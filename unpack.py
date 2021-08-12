import os
import re
import pickle
import pandas as pd
import shutil
from zipfile import ZipFile
from tqdm import tqdm

if __name__ == "__main__":
    print("Creating patient-file mapping")
    zip_files = [x for x in os.listdir() if x.endswith("zip")]
    if not os.path.exists("patient_files.pkl"):
        patients_files = {}
        for zip_file in zip_files:
            with ZipFile(zip_file, 'r') as z:
                # Get list of files names in zip
                contents = z.namelist()
                patients = set([re.search(r"([0-9]+_[0-9]+_[0-9]+)", x).group(1) for x in contents])
                for patient in patients:
                    patients_files[patient] = zip_file
        pickle.dump(patients_files, open("patient_files.pkl", "wb"))
    else:
        with open("patient_files.pkl", 'rb') as f:
            patients_files = pickle.load(f)
    print("Unpacking only annotated slices")
    metadata = pd.read_csv("DL_info.csv")
    for i, row in tqdm(enumerate(metadata.itertuples()), total=len(metadata)):
        filename = row.File_name
        patient_name, png = "_".join(filename.split("_")[:-1]), filename.split("_")[-1]
        zip_file = patients_files[patient_name]

        if not os.path.exists(f"images/{filename}"):
            with ZipFile(zip_file) as z:
                with z.open(f"Images_png/{patient_name}/{png}") as zf, open(f'images/{filename}', 'wb') as f:
                    shutil.copyfileobj(zf, f)

    # with zipfile.ZipFile('/path/to/my_file.apk') as z:
    #     with z.open('/res/drawable/icon.png') as zf, open('temp/icon.png', 'wb') as f:
    #         shutil.copyfileobj(zf, f)