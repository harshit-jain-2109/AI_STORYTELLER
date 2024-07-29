# AI_STORYTELLER

This project is a Story to Video Generator that converts text stories into videos with corresponding images and narrated text. Using a combination of machine learning models and media processing libraries, the application generates relevant images for each sentence in the story, overlays the text on the images, and adds text-to-speech narration to create a cohesive video.
![Flow diagram](https://github.com/user-attachments/assets/180a9c9b-8d9b-4931-bfa7-e40e88bf636d)


## Features

- Text-to-Image Generation: Uses Stable Diffusion to generate images based on the story sentences.
- Text Overlay: Adds the corresponding text on the generated images.
- Text-to-Speech: Converts the text into speech using Google Text-to-Speech (gTTS).
- Video Creation: Combines images and audio into a video clip for each sentence and concatenates them into a final video.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/harshit-jain-2109/AI_STORYTELLER.git
   cd AI_STORYTELLER
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   
## Usage

- You can run the application by executing the script::
   ```
   python gradio.py
   ```
- This will launch a Gradio interface in your web browser where you can enter your story text and generate a video.

## Generating a Video
1. Enter your story in the text box.
2. Click the "Submit" button.
3. Wait for the video to be generated and download it once it's ready.
4. Here is the sample generated videos.


https://github.com/user-attachments/assets/946f1a89-3950-4319-8af8-84ca5abb7a15

https://github.com/user-attachments/assets/25ef5f33-d539-4f46-a7ad-8d01af4c48ee




# Code Explanation
## Imports and Setup
- The script starts by importing necessary libraries and setting up configurations for the Stable Diffusion model.

## Image Generation
- The generate_image function generates an image from a given text prompt using the Stable Diffusion model. The draw_text_on_image function overlays text on the generated images.

## Story Processing
- The process_story function processes the story by splitting it into sentences, generating images and audio for each sentence, and creating individual video clips. It then concatenates all clips into a final video.

## Gradio Interface
- The Gradio interface is set up using the gr.Interface class, which defines the input as text and output as video.

## Example
Here's an example of how to use the generator:

1. Run the application.
2. Enter a story such as:
   ```
   Once upon a time, in a faraway land, there was a small village. The villagers were kind and hardworking. One day, a mysterious traveler arrived in the village.
   ```
3. Click "Submit" and wait for the video to be generated.

## Contributing

Contributions to this project are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with descriptive commit messages
4. Push your changes to your fork
5. Submit a pull request to the main repository

Please ensure that your code adheres to the project's coding standards and includes appropriate tests and documentation.

## Acknowledgements

- Thanks to Stability AI for the Stable Diffusion model.
- Thanks to Hugging Face for their model hub and libraries.
- Thanks to Google for the Text-to-Speech API.


## Contact

For questions or support, please open an issue on the GitHub repository or contact harshitjain3349@gmail.com directly.
