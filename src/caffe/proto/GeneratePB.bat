set PROTOC=../../../../../install/vsprojects/Release/protoc.exe

echo caffe.pb.h is being generated
"%PROTOC%" --cpp_out="./" caffe.proto

copy /y "caffe.pb.h" "..\\..\\..\\..\\..\\install\\include\\caffe\\caffe.pb.h"
move  "caffe.pb.h" "..\\..\\..\\..\\..\\Caffe-prefix\\src\\Caffe-build\\include\\caffe\\caffe.pb.h"
move "caffe.pb.cc" "..\\..\\..\\..\\..\\Caffe-prefix\\src\\Caffe-build\\include\\caffe\\caffe.pb.cc"
