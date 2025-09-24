
import requests

import requests
import os
import json


cse_ids = ["31d13999e718d4eeb", "267f49c3a6e824c99", "93c4dcc4cc7b34496", "15cf52d8a88df4c41"]
api_keys = [ "AIzaSyAuV_1nKaX7-lNiT_7hTIrsjhWaPCfg5Gs", "AIzaSyBeRDsjLL0gJBmZ9wlpIhsEB01AKdc63C4", "AIzaSyCoCt5rtHruQCoOXDgoxmeTAUCDf1JsmoQ", "AIzaSyDRTadTICNufwcynhHzWt9XFbajZ0Plubc"]

def search_linkedin_profiles(query, api_key, cse_id, row_index, image_download=True, pp_name=None, output_json="datasets/profile_picture_data/profile_picture_results.json"):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "cx": cse_id,
        "key": api_key,
        "searchType": "image",
        "num": 1,
    }
    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code != 200:
        print(f"Error: {response.status_code} - {data.get('error', {}).get('message', 'No images found.')}")
        return 1
    
    if "items" not in data or not data["items"]:
        print("No images found for the query.")
        return

    os.makedirs("datasets/images", exist_ok=True)

    for _, item in enumerate(data["items"]):
        img_url = item["link"]

        if image_download:
            img_data = requests.get(img_url).content
            with open(pp_name, "wb") as f:
                f.write(img_data)
                print(f"Downloaded: {pp_name}")

        result = {
            "url": img_url,
            "path": pp_name if image_download else None,
            "index": row_index
        }
    # Save JSON results
    with open(output_json, "r") as f:
        results = json.load(f) if os.path.exists(output_json) else []
    with open(output_json, "w") as f:
        results.append(result)
        json.dump(results, f, indent=4)
        print(f"Saved results to {output_json}")
    return 0

