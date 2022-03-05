FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
RUN pip install pytorch-lightning==1.5.10
WORKDIR /nccl-test
COPY ./* ./
RUN apt update
RUN apt-get install -y build-essential
RUN make
CMD /build/all_reduce_perf -b 8 -e 128M -f 2 -g 2