[![LinkedIn][linkedin-shield]][linkedin-url]
<!--
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


[![Github][github-shield]][github.com/zoumson?tab=repositories]
[![Stack Overflow][stackoverflow-shield]][stackoverflow.com/users/11175375/adam]
[![Leetcode][leetcode-shield]][eetcode.com/Hard_Code/]
-->
## Counting cars passing to a defined line
![image](https://user-images.githubusercontent.com/38358621/123052500-779c0180-d435-11eb-9299-224eb9ff9daf.png)

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#file-structure">Files Structure</a>
      <ul>
        <li><a href="#folders">Folders</a></li>
        <li><a href="#entire-files-structure">Entire Files Structure</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->
Counting cars passing to a defined line.
<!--Built with -->
### Built With

<br>

* [opencv](https://opencv.org/)
* [cmake](https://cmake.org/)

<br>

## File Structure

### Folders

* [include/](include/): c++ header files.
* [src/](src/): c++ definitions.


### Entire Files Structure 

```
.
├── CMakeLists.txt
├── include
│   ├── Blob.h
│   ├── ConstantsTracking.h
│   ├── DrawWithBlob.h
│   └── UseBlob.h
├── README.md
├── resource
│   └── video
│       └── CarsDrivingUnderBridge.mp4
└── src
    ├── demo
    │   ├── CMakeLists.txt
    │   └── demo.cpp
    └── lib
        ├── blob
        │   ├── Blob.cpp
        │   ├── DrawWithBlob.cpp
        │   └── UseBlob.cpp
        └── CMakeLists.txt

7 directories, 13 files

```


<!-- GETTING STARTED -->
## Getting Started

This is a sample code of how you may use  the opencv libs to count cars passing to a difined line.
To get a local copy up and running follow these simple steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* cmake
  ```sh
  sudo apt-get install cmake
  ```

 * opencv4
 ```sh
 sudo su
 ```
 ```sh
git clone https://github.com/zoumson/OpencvInstall.git     \
&& cd OpencvInstall && chmod +x install.sh && ./install.sh
 ```
### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/zoumson/CarCounting.git
   ```
2. Go to the project directory source
   ```sh
   cd CarCounting
   ```
3. Create empty directories 
   ```sh
   mkdir build &&  mkdir bin && mkdir lib
   ```
5. Generate the exectutable and move it to `bin`
   ```sh
   cd build && cmake .. && make -j4 && cd ..
   ```

<!-- USAGE EXAMPLES -->
### Usage

2. Run the executable 
 ```sh
   ./bin/demo 
```

6. Back to the initial file structure configuration
   ```sh
   rm -r bin build lib 
   ```
<!-- ROADMAP -->
## Roadmap

All the headers files are well docummented, read through the comments

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Adama Zouma - <!-- [@your_twitter](https://twitter.com/your_username) -->- stargue49@gmail.com

Project Link: [https://github.com/zoumson/CarCounting](https://github.com/zoumson/CarCounting.git)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Google](https://www.google.com/)
* [Stack Overflow](https://stackoverflow.com/)
* [Github](https://github.com/)




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: linkedin.com/in/adama-zouma-553bba13a
[product-screenshot]: images/screenshot.png

