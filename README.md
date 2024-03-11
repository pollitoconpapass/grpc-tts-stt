# gRPC Text-to-Speech and Speech-to-Text Example

### Steps
1. Install the requirements

        pip install -r requirements.txt


2. Run the qhali.proto

        python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. qhali.proto


3. Run the Server and Client files in different terminals

        python server.py
        python client.py
