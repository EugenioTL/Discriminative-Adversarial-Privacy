FROM tensorflow/tensorflow:2.12.0-gpu

RUN apt-get update && apt-get -y install \
	graphviz \
	git \
	libgl1
RUN pip install --upgrade pip
RUN git clone https://github.com/omerio/graphviz-server.git /opt/graphviz-server

# Create a working directory
WORKDIR /app
# Install extras
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
CMD ["bash"]
WORKDIR /exp


# Enable jupyter
RUN mkdir /.local
RUN chmod -R 777 /.local