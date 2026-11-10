# AgapAI restore guide

This guide restores the repository plus the model file and MongoDB export that
are stored separately. The MongoDB export contains personal data. Share it only
with Annie and group members authorized to access it; do not put it in Git.

## Prerequisites

- Windows PowerShell.
- Git and Git LFS. The captured versions were Git 2.50.1.windows.1 and Git LFS
  3.7.0.
- Python 3.11. The captured environment used Python 3.11.0.
- Node.js 24 and npm. The captured versions were Node v24.19.0 and npm 11.17.0.
- MongoDB Server 6.0 or newer, with its service running.
- Internet access for package installation and first-time Hugging Face model
  downloads. The dependency install and two model downloads exceed 500 MB; get
  approval before starting those downloads.

## Clone and restore Git LFS files

```powershell
git clone https://github.com/annierosemamitag23-lang/ysafiles.git C:\AgapAI
Set-Location C:\AgapAI
git lfs pull
```

`heads.pt` is tracked by Git LFS at
`AgapAI\agapai\api\agapai_api\models\agapai_multitask_xlmr\heads.pt`, next
to `meta.json` and beside the `encoder` folder. If it contains the short Git
LFS pointer text instead of tensor data, run `git lfs pull` again.

## Put the separate classifier model in place

Download `model.safetensors` from Drive or USB. Model location:

`AgapAI\agapai\api\agapai_api\models\agapai_multitask_xlmr\encoder\model.safetensors`

See [Where the backup files are stored](#where-the-backup-files-are-stored) for
the download link and saved filename.

Verify the model file before using it:

| File | Expected size (bytes) | SHA256 |
| --- | ---: | --- |
| `encoder\model.safetensors` | 1112197096 | `E8B863AB898B405FA5E7AD25B39177402413DA173FF15EECCEEB6D91500433BF` |

The restore script prompts for the source path if the file is missing. It
verifies the source before copying and does not replace a mismatched existing
target.

## Python requirements

| File | What it contains | Restore decision |
| --- | --- | --- |
| `AgapAI\agapai\api\requirements.txt` | API-facing dependency names with a mix of unpinned, minimum, and exact version constraints. | Reference only; it does not capture the complete installed environment. |
| `.dependency-setup\requirements-compatible.txt` | A broader compatibility set, including data-science packages and older compatibility pins such as Transformers 4.36.2 and protobuf below 5. | Reference only; its package set and versions differ from the working venv. |
| `restore-kit\requirements-frozen.txt` | A 302-line `pip freeze` snapshot from a Python 3.11 venv configured with `include-system-site-packages = true`; it includes packages from the host Python installation. | Record only; do not install from this whole-machine snapshot. |
| `restore-kit\requirements-restore.txt` | The project imports and test dependencies, pinned with their compatible transitive dependencies at the working versions; the set was resolved with pip's report mode. | **Use this for installation.** |

The earlier dry-run failure came from unrelated host packages included by
system-site-packages: MediaPipe requires `protobuf` below 5, while the Google
AI dependencies in that host environment require `protobuf` 5.29.6 or newer.
The project restore set excludes those unused packages, removing that conflict
without changing the project requirements or the working environment.

The restore set does not need the machine-local `deepface` or OpenCV wheel
references in the whole-machine snapshot. It contains the project runtime and
test imports, including transitive dependencies, with working-environment
version pins. `restore.ps1` installs this file with `--no-cache-dir` and asks
before starting the dependency download.

To run the automated restore after cloning, use:

```powershell
.\restore-kit\restore.ps1
```

The script creates `AgapAI\agapai\.venv` if it is absent, installs
`restore-kit\requirements-restore.txt`, runs `npm ci` if `node_modules` is
absent, prepares NLTK and Hugging Face resources, checks the model and LFS
file, and offers to import the MongoDB export. It never removes files. It
skips `npm ci` if `node_modules` already exists because npm may replace that
directory.

## MongoDB and the private data export

Install MongoDB Server 6.0 or newer and start its service. The API's default
MongoDB host is `localhost:27017`; it uses database `AgapAI_Database_Final`.
The current Windows service is named `MongoDB`. The API uses `MONGO_URI` when
set and otherwise connects to the local server.

The export is stored separately on Drive or USB; see
[Where the backup files are stored](#where-the-backup-files-are-stored) for its
download link. The export contains personal data, including handles and post
text, and must be shared only with Annie and group members authorized to access
it. Keep the export folder outside the repository.

The manifest lists every collection in `AgapAI_Database_Final`:

| Collection | Expected documents | Export file size (bytes) | SHA256 |
| --- | ---: | ---: | --- |
| `events` | 1 | 1380 | `cfc48290b148e97d23f5dca94c4c7e049b3f373c2719acc7c1336ff9d339ebc9` |
| `posts` | 60 | 234887 | `8cfc50e535551bab2355939b7ae0a87958ec6f0399510943f09994f63609e78d` |
| `users` | 22 | 6983 | `b5afb9b3cfce930cb32a6858854d319f30bb9266486b77358772a810e4eb20dc` |

There was no `raw_posts` collection in the inspected database. The export files
are Extended JSON arrays, one file per collection, with checksums in
`manifest.json`. To create and check an export from a live database, run these
scripts with the project venv:

```powershell
& .\AgapAI\agapai\.venv\Scripts\python.exe .\restore-kit\export_mongo.py --out 'D:\agapai_mongo_export'
& .\AgapAI\agapai\.venv\Scripts\python.exe .\restore-kit\verify_mongo_export.py --out 'D:\agapai_mongo_export'
```

The exporter reads collection names and documents only; it prints collection
names and counts. The verifier parses every file, checks live counts and `_id`
sets, and checks file sizes and SHA256 values. Do not export `admin`, `config`,
or `local`.

To restore data, first ensure the target collections are empty. The importer
refuses to overwrite any collection that already contains documents, imports
every collection in the manifest, and compares the final counts:

```powershell
& .\AgapAI\agapai\.venv\Scripts\python.exe .\restore-kit\import_mongo.py --export 'D:\agapai_mongo_export'
```

Expected counts after import are `events` 1, `posts` 60, and `users` 22.

## Where the backup files are stored

1. **Classifier model:** [Download `model_backup.safetensors` from Google
   Drive](https://drive.google.com/file/d/1M1NkZm9070qSV3g0cfC87f3hOP3Rxmnc/view?usp=drive_link).
   It is 1,112,197,096 bytes. Save it as `model.safetensors` in
   `AgapAI\agapai\api\agapai_api\models\agapai_multitask_xlmr\encoder`.
   Verify that its SHA256 matches `restore-kit\model-checksum.txt`.

2. **MongoDB export:** [Download `agapai_mongo_export.zip` from Google
   Drive](https://drive.google.com/file/d/1x9wR-tzoy3Sz7GzHkLtP_YbCV2CyPolx/view?usp=drive_link).
   Unzip it. The extracted `agapai_mongo_export` folder contains
   `manifest.json` and an `AgAPAI_Database_Final` subfolder with `events.json`,
   `posts.json`, and `users.json`. Pass the extracted folder containing
   `manifest.json` to `restore-kit\import_mongo.py --export <unzipped-folder>`;
   do not pass the `AgAPAI_Database_Final` subfolder. Verify the counts are 1,
   60, and 22.

3. **Optional group Git history bundle:** [Download
   `agapai_group_history.bundle` from Google
   Drive](https://drive.google.com/file/d/1-njfRhfXNf-LqO157cyp2cHx7W8XR-7t/view?usp=drive_link).
   Restore it with `git clone agapai_group_history.bundle <folder>`, replacing
   `<folder>` with the destination path. It contains the group's commits up to
   the backup time and does not include the old model versions.

4. **Full group history folder:** [Open the `.git_group_backup` folder in
   Google Drive](https://drive.google.com/drive/folders/1JF4jjjtkAJjjFwedBnAEqejwBukKLzpm?usp=drive_link).
   This is a separate backup of the group's old Git history and LFS objects,
   including older model files not included in the bundle above. It is optional
   for restoring the app from this repository, but may be needed to recover the
   group's older history or model versions.

The MongoDB export, group history bundle, and full group history folder may
contain other people's account handles, post text, credentials, or model files.
Share them only with Annie and group members authorized to access them.

## Environment variables

Set environment variables locally or in `AgapAI\agapai\api\.env`. Do not
commit `.env` or real credentials. The real Bluesky credentials are secrets and
are not included in this restore kit; whoever restores the project must supply
`BLUESKY_HANDLE` and `BLUESKY_PASSWORD`.

Names used by the project:

- `MONGO_URI`
- `AGAPAI_MODEL_DIR`
- `AGAPAI_DEVICE`
- `AGAPAI_BATCH_SIZE`
- `AGAPAI_TEST_MONGO_URI`
- `BLUESKY_HANDLE`
- `BLUESKY_PASSWORD`
- `BLUESKY_LOGIN_ATTEMPTS`
- `BLUESKY_LOGIN_RETRY_DELAY`

`AGAPAI_MODEL_DIR` can point to the classifier model directory containing
`meta.json`, `heads.pt`, and `encoder`. The default is the model folder inside
`agapai_api\models\agapai_multitask_xlmr`.

## NLP resources and model revisions

NLTK resources used by the API are `wordnet` and `omw-1.4`. The restore script
downloads them with NLTK's downloader.

| Hugging Face model | Code usage | Captured revision | Approximate cached size |
| --- | --- | --- | ---: |
| `ljvmiranda921/tl_calamancy_md` | `snapshot_download`, explicitly requests revision `0.2.0` | `0.2.0` | Varies by selected files |
| `facebook/nllb-200-distilled-600M` | `from_pretrained`, default revision `main` | cached `main`: `f8d333a098d19b4fd9a8b18f94170487ad3f821d`; another cached PR revision: `a3e77be725cf30383f1faeb8d2f0b0ea98ac554e` | about 2.31 GiB |
| `facebook/bart-large-mnli` | `from_pretrained`, default revision `main` | cached `main`: `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` | about 1.52 GiB |

The restore script can prefetch the recorded revisions after asking before each
large download. The application calls `from_pretrained` without a revision for
NLLB and BART, so its first runtime load can resolve the current `main` branch
over the internet if that revision is not already cached. The recorded hashes
identify the local cache snapshots observed during this backup.

## Start and verify the app

Start the API in one PowerShell window:

```powershell
Set-Location .\AgapAI\agapai\api
..\.venv\Scripts\python.exe -m uvicorn agapai_api.app:app --reload --host 127.0.0.1 --port 8000
```

Start the dashboard in another window:

```powershell
Set-Location .\AgapAI\agapai\dashboard
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

For verification, with MongoDB running and environment variables set:

```powershell
Set-Location .\AgapAI\agapai\api
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
Set-Location ..\dashboard
node --test src/spatiotemporal.test.js
npm run lint
npm run build
```

The dashboard package does not define an npm `test` script; its test file is
run with Node's test runner as shown above.

The `test_map_and_anchors_prefer_specific_place_over_province` test in
`test_ner_pipeline.py` also fails in the original environment: it expects
`Pardo` but receives `Cebu`.

## Troubleshooting

- **`heads.pt` is a tiny text file:** it is an LFS pointer. Install Git LFS and
  run `git lfs pull` from the repository root.
- **Classifier model missing:** obtain `model.safetensors` from the Drive/USB
  placeholder above, put it under `encoder`, and verify its size and SHA256.
- **Wrong Python version:** install Python 3.11 and rerun the restore script; do
  not reuse a venv created with another Python version.
- **Hugging Face model downloads fail or differ:** ensure internet access. The
  first runtime loads can download model files; NLLB and BART are multi-gigabyte
  resources. Approve large downloads before starting them.
- **MongoDB import refuses:** the importer found documents in at least one
  target collection. Back up and review that local database separately; the
  importer will not overwrite it.
- **MongoDB is unavailable:** install MongoDB 6.0 or newer, start its service,
  and check the local `MONGO_URI` setting.
- **Python package resolution reports a conflict:** confirm that the restore
  used `restore-kit\requirements-restore.txt`, not the whole-machine
  `requirements-frozen.txt` snapshot.

## How this backup was made

The saved PowerShell log shows the outer repository was initialized at
`C:\AgapAI`, while the university project already had its own nested Git
history. That history was retained by renaming the inner `.git` directory to
`.git_group_backup` and excluding it from the outer repository; the project
files remain under `AgapAI\`. The outer `main` branch was then pushed to the
`ysafiles` repository.

The old group-history copy is optional for restoring the app. Its local path at
the time of this backup was `C:\AgapAI\AgapAI\.git_group_backup`; it is now
stored in the [`.git_group_backup` Google Drive folder](https://drive.google.com/drive/folders/1JF4jjjtkAJjjFwedBnAEqejwBukKLzpm?usp=drive_link).
