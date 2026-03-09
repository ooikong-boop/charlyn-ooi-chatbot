# Directive: Data Ingestion & Transcription

## Goal
To process the local video training library, extract the audio from the `.mp4` files, and transcribe it into structured Markdown files for the AI chatbot's knowledge base.

## Inputs
The user has provided the following network drive locations containing the training videos and documentation:

1. **Accelerator Program**: `N:\1.Video Work\Accelerator Program` 
   - *Rule*: ONLY process files in this root directory. Ignore all subfolders.
2. **Momentum Club**: `N:\1.Video Work\Momentum Club`
   - *Rule*: Process all files in this directory AND its subfolders.
3. **Leads on Autopilot**: `N:\1.Video Work\Leads on Autopilot`
   - *Rule*: Process all files in this directory AND its subfolders.
4. **Program Notes**: `N:\1.Current Businesses (S3)\Program Notes`
   - *Rule*: Process all documentation (PDFs, text) in this directory AND its subfolders.
5. **Milestone Guide**: `N:\1.Current Businesses (S3)\Program Notes\Guide\Table 1-Grid view.csv`
   - *Rule*: This CSV acts as the master pathway for guiding users through the programs.

## Processing Steps

**Step 1. Audio Extraction**
- Iterate through the defined directories based on the inclusion rules above.
- For every `.mp4` file, use `ffmpeg` (via Python `subprocess` or `moviepy`) to extract the audio to a temporary `.mp3` file in the `.tmp/` directory.

**Step 2. Transcription**
- Pass the temporary `.mp3` file to the `OpenAI Whisper API`.
- Retrieve the text transcription.

**Step 3. Markdown Formatting**
- Create a `.md` file for each video in a new folder called `knowledge_base/`.
- The `.md` file should include YAML frontmatter with metadata:
  - `Source Program`: (e.g., Accelerator Program)
  - `File Name`: (original video name)
- The body of the `.md` file should contain the chunked transcript text.

**Step 4. Cleanup**
- Delete the temporary `.mp3` file from `.tmp/` to save disk space.

## Outputs
- A `knowledge_base/` directory containing markdown files for every video and document, ready to be ingested into a Vector Database (RAG pipeline).
- Logs capturing any failed transcriptions or unreadable files.

## Edge Cases to Handle
- **Network Interruptions**: The `N:\` drive is a network location. Ensure file paths are properly escaped and the script handles temporary disconnections gracefully.
- **Large Files**: The Whisper API has a 25MB limit on audio files. The script must split the `.mp3` audio internally into `<25MB` segments before sending it to the API, and then stitch the transcribed text back together.
