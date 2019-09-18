### kinect_acquisition

To run the kinect_acquisition launch follow the following instructions

**Libfreenect install**

```javascript

cd  ~
git clone https://github.com/OpenKinect/libfreenect.git
cd libfreenect
mkdir build
cd build
cmake -L ..
make
sudo make install

```

**Freenect_launch**

```javascript

cd ~/arlobot_ws/src/arlobot
git clone https://github.com/ros-drivers/freenect_stack.git
cd ..
catkin_make

```
