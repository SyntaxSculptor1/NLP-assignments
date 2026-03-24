CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]

def get_model_settings() -> dict[str, dict[str, bool | float]]:
    """
    Get the model settings.

    Returns:
        dict: The model settings.
    """
    settings = {}
    
    for input_field in ["Headline", "HeadlineDescription"]:
        for noise in [0.25, 0.75, 1.00]:
            settings[f"Transformer-{input_field}-{noise}"] = {
                "description": input_field == "HeadlineDescription",
                "noise": noise,
            }
            
    return settings