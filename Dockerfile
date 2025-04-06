FROM python:3.11-slim

# Copying project files to the image
WORKDIR /m2
COPY . .

# Installing requirements
RUN pip install -r requirements.txt

# Installing m2_utilities as an editable package
RUN pip install -e .

# Start the Jupyter notebooks on port 8888
WORKDIR /m2
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", \
    "--allow-root", "--NotebookApp.token=''", "--notebook-dir=/m2/notebooks"]
