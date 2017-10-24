# ThaiLetterRecognition
A small programm to localize and identify Thai text in natural images using OpenCv and TensorFlow.

## Goal

### Main

Input natural image with Thai language text and output text in unicode

### Optional

Port code to Android and add Google Translator API to translate text on image.

## Roadmap

1. Localize letters in image (OpenCV)

    * [x] Find letter candidates via MSER
  
    * [x] Validate letter candidates via Stroke Width Transform
  
    * [x] Form lines of text from letter candidates
2. Identify letters

    * [ ] Crop letters from images using already working program
  
    * [ ] Annotate letter images
  
    * [ ] Finetune object recognition deep neural network with letter crop outs
  
3. Complete recognition workflow

## Project Status

Tune performance of working features. Code clean up complete.
