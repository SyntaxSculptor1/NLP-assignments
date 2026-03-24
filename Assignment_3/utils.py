CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]

def get_model_settings():
    settings = {}
    
    for input_field in ["Headline", "HeadlineDescription"]:
    #for input_field in ["Headline"]:
        #for noise in [0.25, 0.75, 1.00]:
        for noise in [1.00]:
            settings[f"Transformer-{input_field}-{noise}"] = {
                "description": input_field == "HeadlineDescription",
                "noise": noise,
            }
            
    return settings