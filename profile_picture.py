from googlesearch import search
import requests
from bs4 import BeautifulSoup
import os

def find_linkedin_profile(full_name, company, location):
    query = f'site:linkedin.com/in/ "{full_name}" "{company}" "{location}"'
    # Get the first search result
    for url in search(query, num_results=1):
        return url
    return None

def download_first_image(url, save_path='linkedin_photo.jpg'):
    # Fetch the page content
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Failed to retrieve page")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    print(soup.prettify())
    return ()
    # Find the first image (LinkedIn profile photos usually have meta og:image)
    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.get("content"):
        image_url = og_image["content"]
    else:
        # Fallback to first image tag
        img_tag = soup.find("img")
        if img_tag and img_tag.get("src"):
            image_url = img_tag["src"]
        else:
            print("No image found.")
            return
    print("Image URL:", image_url)
    # Download the image
    return ()

    img_data = requests.get(image_url, headers=headers).content
    with open(save_path, 'wb') as f:
        f.write(img_data)
    print(f"Image saved to {save_path}")

# Example usage site:linkedin.com/in/ "Mounika Mungamuri" "Infosys" "Hyderabad, Telangana, India"
full_name = "Mounika Mungamuri"
company = "Infosys"
location = "Hyderabad, Telangana, India"

profile_url = find_linkedin_profile(full_name, company, location)
print("LinkedIn Profile URL:", profile_url)
#download_first_image(profile_url, save_path='datasets/images/{full_name}_{company}_{location}.jpg')

