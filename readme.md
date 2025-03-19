# Master's degree Thesis

### Guide on how to run the app
To successfully run the code please follow the next steps:

1. Clone the repository

Inside a terminal:

2. Navigate to the Back-end dir and run `python3 manage.py migrate`.
3. Navigate to front-end dir and run `npm install react-scripts --save`. This will create the node modules needed.
4. In the same dir run `npm run build` to build the front end so it can be used by the server.
5. Navigate back to the Back-end dir and run `python3 manage.py runserver --noreload` (to avoid running setup 2 times)


#### Notes for me...

For my computer starts with: /usr/local/bin/python3.12 manage.py runserver --noreload       

Remember to add proxy: "http://localhost:8000/" under ' "private": true, ' in package.json in front-end dir to run with django and delete that when developing the front-end so we skip the model init.


remove files that where pushed to origin before added to git ignore and noticed afterwards: git rm -r --cached node_modules
