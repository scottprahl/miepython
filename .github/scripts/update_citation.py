import json
import requests
import yaml

# Replace with your username and repo name
USERNAME = "scottprahl"
REPO = "miepython"

# Fetch latest release date
response = requests.get(f"https://api.github.com/repos/{USERNAME}/{REPO}/releases/latest")
release_date = json.loads(response.text)["published_at"].split("T")[0]

# Read the existing CITATION.cff file
with open("CITATION.cff", "r") as f:
    cff_data = yaml.safe_load(f)

# Update the date-released field only if it's different
if cff_data.get("date-released") != release_date:
    cff_data["date-released"] = release_date

    # Save the updated data back to CITATION.cff
    with open("CITATION.cff", "w") as f:
        yaml.dump(cff_data, f)
else:
    print("No change in release date. No update needed.")