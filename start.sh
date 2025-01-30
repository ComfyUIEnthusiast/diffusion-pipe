cp -r -u /importModels/* /models
jupyter notebook --NotebookApp.token=$JUPYTER_TOKEN --NotebookApp.ip='0.0.0.0' --port $JUPYTER_PORT --allow-root & 
wait