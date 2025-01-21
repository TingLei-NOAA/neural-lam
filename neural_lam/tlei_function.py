def closest_hour(hour):
    # Define the set of target hours
    target_hours = [0, 6, 12, 18]
    
    # Convert hour to integer and find the closest target hour
    closest = min(target_hours, key=lambda x: abs(x - hour))
    
    # Format it to always have two digits (e.g., '00', '06')
    return f"{closest:02d}"
