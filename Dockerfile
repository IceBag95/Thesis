# Stage 1: Build React app
FROM node:21 AS react-build

WORKDIR /frontend
COPY front-end/ ./
RUN npm install
RUN npm run build

# Stage 2: Set up Django app
FROM python:3.12-slim

# Build argument for host username (default: dockeruser)
ARG HOST_USER=dockeruser

# Install bash and sudo
RUN apt-get update && apt-get install -y bash sudo

# Set up custom home directory at /home/<user>/app
RUN useradd -m -d /home/$HOST_USER/app -s /bin/bash $HOST_USER && \
    echo "$HOST_USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Define base APP path
ENV APP_HOME=/home/$HOST_USER/app

# Switch to user temporarily
USER $HOST_USER
WORKDIR $APP_HOME/Back-end

# Switch to root to copy requirements
USER root
COPY Back-end/requirements.txt $APP_HOME/Back-end/
RUN chown $HOST_USER:$HOST_USER $APP_HOME/Back-end/requirements.txt

# Install Python dependencies
USER $HOST_USER
RUN pip install --no-cache-dir -r $APP_HOME/Back-end/requirements.txt

# Copy project files
USER root
COPY Back-end/ $APP_HOME/Back-end/
COPY ML/ $APP_HOME/ML/
COPY Dataset/ $APP_HOME/Dataset/
COPY --from=react-build /frontend/build $APP_HOME/front-end/build
#RUN chown -R $HOST_USER:$HOST_USER $APP_HOME
RUN chown -R $HOST_USER:$HOST_USER $APP_HOME
RUN chmod -R u+rw $APP_HOME/Back-end $APP_HOME/Dataset

# Add custom PS1 prompt (DOCKER green badge, styled prompt)
RUN echo 'export PS1="\[\e[48;2;20;240;0m\e[1m DOCKER \e[0m\] \[\e[1;97m\][\[\e[1;38;2;20;240;0m\]\u@\h\[\e[1;97m\]:\[\e[1;34m\]\w\[\e[1;97m\]]\[\e[0m\]\$ "' >> /home/$HOST_USER/app/.bashrc

# Add alias for running app with migrate and runserver
RUN echo "alias runapp='python manage.py migrate && python manage.py runserver 0.0.0.0:8000 --noreload'" >> /home/$HOST_USER/app/.bashrc

# Switch back to final user and working dir
USER $HOST_USER
WORKDIR $APP_HOME/Back-end

# Start Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

