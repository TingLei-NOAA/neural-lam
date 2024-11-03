# Standard library
import datetime as dt
import glob
import os

# Third-party
import numpy as np
import torch

# Local
from . import utils
from .tlei_function import closest_hour


class WeatherDataset(torch.utils.data.Dataset):
    """
    For our dataset:
    N_t' = 65
    N_t = 65//subsample_step (= 21 for 3h steps)
    dim_y = 268
#15km rrfs dim_y=268
    dim_x = 238
#15km rrfs    dim_x = 238
    N_grid = 268x238 = 63784
    d_features = 17 (d_features' = 18)
#15km rrfs d_features=41
    d_forcing = 5
#15km rrfs   d_forcing = 5
    """

    def __init__(
        self,
        dataset_name,
        pred_length=5,
#working        pred_length=4,
        split="train",
        subsample_step=3,
        standardize=True,
        subset=False,
        control_only=False,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), "Unknown dataset split"
        self.sample_dir_path = os.path.join(
            "data", dataset_name, "samples", split
        )

        member_file_regexp = (
            "nwp*mbr000*.npy" if control_only else "nwp*mbr*.npy"
        )
        sample_paths = glob.glob(
            os.path.join(self.sample_dir_path, member_file_regexp)
        )
        self.sample_names = [path.split("/")[-1][4:-4] for path in sample_paths]
# for ecample : nwp_2024050718_mbr002_15km.npy
#        self.sample_names = [path.split("/")[-1][4:-9] for path in sample_paths]
        print("thinkdeb sample_dir_path ",self.sample_dir_path)
        print("thinkdeb sample_name ",*self.sample_names)

        # Now on form "yyymmddhh_mbrXXX"

        if subset:
            self.sample_names = self.sample_names[:50]  # Limit to 50 samples
#        nsample=10
#        self.sample_names = self.sample_names[:nsample]  #cltthinkdeb250 Limit to 50 samples
        self.sample_length = pred_length + 2  # 2 init states
        self.subsample_step = subsample_step
        self.original_sample_length = (
            19 // self.subsample_step
#clt            65 // self.subsample_step
#clt 65 should be replaced by 60 in emc-na case, at least, or 
#it should be variables for different sets like forecast from 01UTC and so on. 
        )  # 21 for 3h steps
        assert (
            self.sample_length <= self.original_sample_length
        ), "Requesting too long time series samples"

        # Set up for standardization
        self.standardize = standardize
        if standardize:
            ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
            self.data_mean, self.data_std, self.flux_mean, self.flux_std = (
                ds_stats["data_mean"],
                ds_stats["data_std"],
                ds_stats["flux_mean"],
                ds_stats["flux_std"],
            )

        # If subsample index should be sampled (only duing training)
        self.random_subsample = split == "train"

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        # === Sample ===
        sample_name = self.sample_names[idx]
        sample_path = os.path.join(
            self.sample_dir_path, f"nwp_{sample_name}.npy"
#cltorg            self.sample_dir_path, f"nwp_{sample_name}.npy"
        )
        print("xxx in getimem weather data")
        print("file is " , sample_path)
        try:
            full_sample = torch.tensor(
                np.load(sample_path), dtype=torch.float32
            )  # (N_t', dim_y, dim_x, d_features')
        except ValueError:
            print(f"Failed to load {sample_path}")

        # Only use every ss_step:th time step, sample which of ss_step
        # possible such time series
        if self.random_subsample:
            subsample_index = torch.randint(0, self.subsample_step, ()).item()
        else:
            subsample_index = 0
        subsample_end_index = self.original_sample_length * self.subsample_step
        sample = full_sample[
            subsample_index : subsample_end_index : self.subsample_step
        ]
        # (N_t, dim_y, dim_x, d_features')

        # Remove feature 15, "z_height_above_ground"
#cltorg        sample = torch.cat(
#cltorg            (sample[:, :, :, :15], sample[:, :, :, 16:]), dim=3
#cltorg        )  # (N_t, dim_y, dim_x, d_features)

        # Accumulate solar radiation instead of just subsampling
#clt        rad_features = full_sample[:, :, :, 2:4]  # (N_t', dim_y, dim_x, 2)
        # Accumulate for first time step
#clt        init_accum_rad = torch.sum(
#clt            rad_features[: (subsample_index + 1)], dim=0, keepdim=True
#clt        )  # (1, dim_y, dim_x, 2)
        # Accumulate for rest of subsampled sequence
        in_subsample_len = (
            subsample_end_index - self.subsample_step + subsample_index + 1
        )
#clt        rad_features_in_subsample = rad_features[
#clt            (subsample_index + 1) : in_subsample_len
#clt        ]  # (N_t*, dim_y, dim_x, 2), N_t* = (N_t-1)*ss_step
#clt        _, dim_y, dim_x, _ = sample.shape
#clt        rest_accum_rad = torch.sum(
#            rad_features_in_subsample.view(
#                self.original_sample_length - 1,
#                self.subsample_step,
#                dim_y,
#                dim_x,
#                2,
#            ),
#            dim=1,
#        )  # (N_t-1, dim_y, dim_x, 2)
#        accum_rad = torch.cat(
#            (init_accum_rad, rest_accum_rad), dim=0
#        )  # (N_t, dim_y, dim_x, 2)
#        # Replace in sample
#clt end of block        sample[:, :, :, 2:4] = accum_rad

        # Flatten spatial dim
        sample = sample.flatten(1, 2)  # (N_t, N_grid, d_features)

        # Uniformly sample time id to start sample from
#cltthink
        init_id = torch.randint(
            0, 1 + self.original_sample_length - self.sample_length, ()
        )
        sample = sample[init_id : (init_id + self.sample_length)]
        # (sample_length, N_grid, d_features)

        if self.standardize:
            # Standardize sample
            sample = (sample - self.data_mean) / self.data_std

        # Split up sample in init. states and target states
        init_states = sample[:2]  # (2, N_grid, d_features)
        target_states = sample[2:]  # (sample_length-2, N_grid, d_features)

        # === Forcing features ===
        # Now batch-static features are just part of forcing,
        # repeated over temporal dimension
        # Load water coverage
        sample_datetime = sample_name[:10]
        water_path = os.path.join(
#            self.sample_dir_path, f"wtr_{sample_datetime}.npy"
            self.sample_dir_path, f"wtr_15km.npy"
        )
        water_cover_features = torch.tensor(
            np.load(water_path), dtype=torch.float32
        ).unsqueeze(
            -1
        )  # (dim_y, dim_x, 1)
        # Flatten
        water_cover_features = water_cover_features.flatten(0, 1)  # (N_grid, 1)
        # Expand over temporal dimension
        water_cover_expanded = water_cover_features.unsqueeze(0).expand(
            self.sample_length - 2, -1, -1  # -2 as added on after windowing
        )  # (sample_len, N_grid, 1)

        # TOA flux
        flux_path = os.path.join(
            self.sample_dir_path,
#cltorg            f"nwp_toa_downwelling_shortwave_flux_{sample_datetime}.npy",
            f"nwp_{sample_datetime}_solar_flux_15km.npy",
        )
#clt only flux files on 00 06, 12, 18 exist
        if not os.path.exists(flux_path):
        # If it doesn't exist, extract the last two characters (the hour part)
            date_part = sample_datetime[:-2]  # The part before the last two digits
            hour_part = int(sample_datetime[-2:])  # The last two digits (e.g., '01', '23')
            new_hour = closest_hour(hour_part)
            new_sample_datetime = date_part + new_hour
            flux_path = os.path.join(
                self.sample_dir_path,
    #cltorg            f"nwp_toa_downwelling_shortwave_flux_{sample_datetime}.npy",
                f"nwp_{new_sample_datetime}_solar_flux_15km.npy",
            )
        
        flux = torch.tensor(np.load(flux_path), dtype=torch.float32).unsqueeze(
            -1
        )  # (N_t', dim_y, dim_x, 1)

        if self.standardize:
            flux = (flux - self.flux_mean) / self.flux_std

        # Flatten and subsample flux forcing
        flux = flux.flatten(1, 2)  # (N_t, N_grid, 1)
        flux = flux[subsample_index :: self.subsample_step]  # (N_t, N_grid, 1)
        flux = flux[
            init_id : (init_id + self.sample_length)
        ]  # (sample_len, N_grid, 1)

        # Time of day and year
        dt_obj = dt.datetime.strptime(sample_datetime, "%Y%m%d%H")
        dt_obj = dt_obj + dt.timedelta(
            hours=2 + subsample_index
        )  # Offset for first index
        # Extract for initial step
        init_hour_in_day = dt_obj.hour
        start_of_year = dt.datetime(dt_obj.year, 1, 1)
        init_seconds_into_year = (dt_obj - start_of_year).total_seconds()

        # Add increments for all steps
        hour_inc = (
            torch.arange(self.sample_length) * self.subsample_step
        )  # (sample_len,)
        hour_of_day = (
            init_hour_in_day + hour_inc
        )  # (sample_len,), Can be > 24 but ok
        second_into_year = (
            init_seconds_into_year + hour_inc * 3600
        )  # (sample_len,)
        # can roll over to next year, ok because periodicity

        # Encode as sin/cos
        # ! Make this more flexible in a separate create_forcings.py script
        seconds_in_year = 365 * 24 * 3600
        hour_angle = (hour_of_day / 12) * torch.pi  # (sample_len,)
        year_angle = (
            (second_into_year / seconds_in_year) * 2 * torch.pi
        )  # (sample_len,)
        datetime_forcing = torch.stack(
            (
                torch.sin(hour_angle),
                torch.cos(hour_angle),
                torch.sin(year_angle),
                torch.cos(year_angle),
            ),
            dim=1,
        )  # (N_t, 4)
        datetime_forcing = (datetime_forcing + 1) / 2  # Rescale to [0,1]
        datetime_forcing = datetime_forcing.unsqueeze(1).expand(
            -1, flux.shape[1], -1
        )  # (sample_len, N_grid, 4)

        # Put forcing features together
        forcing_features = torch.cat(
            (flux, datetime_forcing), dim=-1
        )  # (sample_len, N_grid, d_forcing)

        # Combine forcing over each window of 3 time steps
        forcing_windowed = torch.cat(
            (
                forcing_features[:-2],
                forcing_features[1:-1],
                forcing_features[2:],
            ),
            dim=2,
        )  # (sample_len-2, N_grid, 3*d_forcing)
        # Now index 0 of ^ corresponds to forcing at index 0-2 of sample

        # batch-static water cover is added after windowing,
        # as it is static over time
        forcing = torch.cat((water_cover_expanded, forcing_windowed), dim=2)
        # (sample_len-2, N_grid, forcing_dim)
        print(f"init_state.shape  {init_states.shape} dtype {init_states.dtype}")
        print(f"target_state.shape  {target_states.shape} dtype: {target_states.dtype}")
        print(f"forcing.shape  {forcing.shape} dtype : {forcing.dtype}")
        if torch.isnan(init_states).any() or torch.isinf(init_states).any() :
           print("abnormal values  found  for init_states quit")
           quit()
        if torch.isnan(target_states).any() or torch.isinf(target_states).any() :
           print("abnormal values  found  for target_states quit")
        if torch.isnan(forcing).any() or torch.isinf(forcing).any() :
           print("abnormal values  found  for forcing quit")
           quit()
           quit()

        return init_states, target_states, forcing
