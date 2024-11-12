# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster AS build

# Set the working directory in the container to /app
WORKDIR /app

COPY requirements-flask.txt .

COPY zscaler.cer .

# Change permissions of the .cer file to be readable by all users
RUN chmod a+r zscaler.cer

# Convert the .cer file to .pem
COPY zscaler.cer /usr/local/share/ca-certificates/zscaler.crt

# Update the certificate authorities
RUN update-ca-certificates

# Set the REQUESTS_CA_BUNDLE environment variable
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt


# Install pip packages
COPY requirements-flask.txt .
RUN pip3 install --upgrade pip \
    && pip3 wheel --wheel-dir /wheels -r requirements-flask.txt

# Use a smaller base image for the final stage
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy only the dependencies installation from the 1st stage image
COPY --from=build /wheels /wheels
COPY --from=build /app/requirements-flask.txt .

COPY --from=build /app/zscaler.cer .

# Change permissions of the .cer file to be readable by all users
RUN chmod a+r zscaler.cer

# Convert the .cer file to .pem
COPY zscaler.cer /usr/local/share/ca-certificates/zscaler.crt

# Update the certificate authorities
RUN update-ca-certificates

# Set the REQUESTS_CA_BUNDLE environment variable
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN pip3 install --find-links=/wheels -r requirements-flask.txt

# Copy the rest of the application
COPY . .


# Declare the environment variable
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

ENV PYTHONHTTPSVERIFY=0

# Make port 80 available to the world outside this container
EXPOSE 5100

ENV FLASK_APP=app.py
CMD ["flask", "run", "--host", "0.0.0.0"]