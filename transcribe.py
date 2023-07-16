import os
import torch
import whisperx
import gc

def transcribe_audio(audio_file_name, your_hf_token, model_file_name):
    device = "cuda"
    compute_type = "float16"  # Ändern Sie dies in "int8", wenn der GPU-Speicher knapp ist (kann die Genauigkeit reduzieren)
    batch_size = 16  # Reduzieren Sie dies, wenn der GPU-Speicher knapp ist

    # Setzen Sie die Pfade für die Audiodatei und das Modell
    audio_file_path = os.path.join("audio_files", audio_file_name)
    model_save_path = os.path.join("models", model_file_name)

    # Überprüfen, ob das Modell bereits lokal gespeichert wurde
    if os.path.exists(model_save_path):
        model = torch.load(model_save_path)
    else:
        # Laden Sie das Audiomodell und speichern Sie es lokal
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        torch.save(model, model_save_path)

    # Laden Sie die Audiodatei
    audio = whisperx.load_audio(audio_file_path)

    # Transkribieren Sie die Audiodatei
    result = model.transcribe(audio, batch_size=batch_size)
    print(result["segments"]) # vor der Ausrichtung

    # Löschen Sie das Modell, wenn der GPU-Speicher knapp ist
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Richten Sie die Whisper-Ausgabe aus
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print(result["segments"]) # nach der Ausrichtung

    # Löschen Sie das Modell, wenn der GPU-Speicher knapp ist
    del model_a
    gc.collect()
    torch.cuda.empty_cache()

    # Weisen Sie Sprecherlabels zu
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=your_hf_token, device=device)

    # Fügen Sie die minimale/maximale Anzahl der Sprecher hinzu, wenn bekannt
    # diarize_segments = diarize_model(audio_file_path, min_speakers=min_speakers, max_speakers=max_speakers)
    diarize_segments = diarize_model(audio_file_path)
    
    result = whisperx.assign_word_speakers(diarize_segments, result)

    print(diarize_segments)
    print(result["segments"]) # Segmente sind jetzt Sprecher-IDs zugewiesen

    return result
