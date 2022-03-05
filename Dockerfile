FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
#FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
RUN pip install pytorch-lightning==1.5.10
WORKDIR /nccl-test
COPY ./* ./
RUN apt update
RUN apt-get install -y build-essential
#RUN make CUDA_HOME=/opt/conda/pkgs/pytorch-1.10.0-py3.7_cuda11.3_cudnn8.2.0_0/lib/python3.7/site-packages/torch/cuda NCCL_HOME=/opt/conda/pkgs/pytorch-1.10.0-py3.7_cuda11.3_cudnn8.2.0_0/lib/python3.7/site-packages/caffe2/contrib/nccl
#RUN make CUDA_HOME=/opt/conda/pkgs/pytorch-1.6.0-py3.7_cuda10.1.243_cudnn7.6.3_0/lib/python3.7/site-packages/torch/cuda NCCL_HOME=/opt/conda/lib/python3.7/site-packages/caffe2/contrib/nccl
CMD ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8sh