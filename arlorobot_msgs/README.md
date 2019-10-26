# ARLO ROBOT MESSAGES

**Structure of the file Detection.msg**

- Class object
- bb_x1 (Corner top left: x coordinates)
- bb_y1 (Corner top left: y coordinates)
- bb_x2 (Corner bottom right: x coordinates)
- bb_y2 (Corner bottom right: y coordinates)

**Structure of the file DetectionArray.msg**

- Header header
- Detection[] DetectionArray

**Structure of the file LandMark.msg**

- Class object
- X_ij (Position X of the object)
- Y_ij (Position Y of the object)
- Z_ij (Position Z of the object)

**Structure of the file LandMarkArray.msg**

- Header header
- LandMark[] LandMarkArray
