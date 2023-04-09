import os
import firebase_admin 
from firebase_admin import credentials, storage

cred = credentials.Certificate('./clasificador-figuras.json')
firebase_admin.initialize_app(cred, 
                            {'storageBucket': 'clasificador-figuras.appspot.com'})
bucket = storage.bucket()

blobs = bucket.list_blobs()

os.makedirs('images/', exist_ok=True)

for blob in blobs:
    if blob.content_type.startswith('image/'):
        filename = os.path.join('images/', blob.name.split('clasificador/')[1])
        blob.download_to_filename(filename)