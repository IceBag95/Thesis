# Master's degree Thesis

### Guide on how to run the app
To successfully run the code please follow the next steps:

1. Clone the repository

Inside a terminal:

2. Navigate to the Thesis folder. 
3. From there execute `docker compose build`.
4. Then execute `docker run -it -p 8000:8000 thesis /bin/bash`.
5. When inside the container execute `runapp`.

The app runs on "http://localhost:8000/predict". You can access this URL through any browser.

NOTE: You can use the docker desktop app as well if available on your OS. The container runs on Debian and it's pretty barebones so feel free to use apt to download any tools you want 

