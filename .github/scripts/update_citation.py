"""This updates the zenodo citation."""
import json
import requests
import yaml

# Replace with your username and repo name
USERNAME = "scottprahl"
REPO = "miepython"

# Fetch latest release date
response = requests.get(f"https://api.github.com/repos/{USERNAME}/{REPO}/releases/latest")
release_info = json.loads(response.text)
release_date = release_info["published_at"].split("T")[0]
version = release_info["tag_name"]

# Read the existing CITATION.cff file
with open("CITATION.cff", "r") as f:
    cff_data = yaml.safe_load(f)

# Create a flag to track if any change is made
changed = False

# Update the date-released field only if it's different
if cff_data.get("date-released") != release_date:
    cff_data["date-released"] = release_date
    changed = True

# Update the version field only if it's different
if cff_data.get("version") != version:
    cff_data["version"] = version
    changed = True

# Save the updated data back to CITATION.cff only if there was a change
if changed:
    with open("CITATION.cff", "w") as f:
        yaml.dump(cff_data, f)
else:
    print("No change in release date or version. No update needed.")
