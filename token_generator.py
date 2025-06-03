from google_auth_oauthlib.flow import InstalledAppFlow

def generate_refresh_token():
    # Replace these values with your actual credentials from the client_secrets.json file
    client_config = {
        "installed": {
            "client_id": "810447952617-0ummuipp61bkujk96qsoig4g966l4b3i.apps.googleusercontent.com",
            "project_id": "thelenders-app",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": "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",
            "redirect_uris": [
            "http://localhost"
            ]
        }
    }

    scopes = ["https://www.googleapis.com/auth/adwords"]
    flow = InstalledAppFlow.from_client_config(client_config, scopes)
    credentials = flow.run_local_server(port=8080, prompt='consent')
    
    print(f"\nYour refresh token is:\n{credentials.refresh_token}")

generate_refresh_token()
