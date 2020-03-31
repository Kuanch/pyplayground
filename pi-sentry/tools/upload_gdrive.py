import os
import pickle
import argparse

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ['https://www.googleapis.com/auth/drive']
CLIENT_SECRET_FILE = './credentials.json'
APPLICATION_NAME = 'Drive API Python Quickstart'


def create_service(creds):
    service = build('drive', 'v3', credentials=creds)

    return service


def set_cred(token_path):
    creds = None
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print('credential is out of date, updating...')
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_console()

        with open(token_path, 'wb') as token:
            print('dumping credential')
            pickle.dump(creds, token)

    return creds


def upload_file(service, file_path, mimetype=None, folder_id=None):
    file_name = os.path.basename(file_path)
    parents = [folder_id] if folder_id is not None else None
    file_metadata = {'name': file_name, 'parents': parents}
    media = MediaFileUpload(file_path)
    file_size = media.size()

    file_id = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    if file_id:
        print('Upload {} success, total {} byte'.format(file_name, file_size))


def main(args):
    creds = set_cred(args.token_path)
    service = create_service(creds)
    upload_file(service, args.file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path')
    parser.add_argument('--token_path')
    args = parser.parse_args()
    main(args)
