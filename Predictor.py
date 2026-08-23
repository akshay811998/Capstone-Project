import pandas as pd
import pickle
import re

best_rf = pickle.load(open("counterfeit_model.sav", "rb"))
sc = pickle.load(open("scaler.sav", "rb"))
selected_cols_list = pickle.load(open("selected_features.sav", "rb"))

def is_suspicious_brand(name):
    if not isinstance(name, str) or name.strip() == '':
        return 0
    if re.search(r'\d', name):
        return 1
    if re.search(r'[^A-Za-z0-9\s]', name):
        return 1
    return 0

def is_counterfeit(price, seller_reviews, product_images, description_length,
                    shipping_time_days, domain_age_days, views, purchases,
                    warranty_months, brand_name):

    brand_suspicious = is_suspicious_brand(brand_name)

    user_input = pd.DataFrame([[price, seller_reviews, product_images, description_length,
                                 shipping_time_days, domain_age_days, views, purchases,
                                 warranty_months, brand_suspicious]],
                               columns=selected_cols_list)

    user_input_scaled = sc.transform(user_input.values)
    prediction = best_rf.predict(user_input_scaled)
    prediction_proba = best_rf.predict_proba(user_input_scaled)

    return {
        "prediction": "COUNTERFEIT" if prediction[0] == 1 else "GENUINE",
        "counterfeit_probability": round(prediction_proba[0][1] * 100, 2),
        "brand_flagged_suspicious": bool(brand_suspicious)
    }