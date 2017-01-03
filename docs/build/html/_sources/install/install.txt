Installation Instructions
=========================

Dependencies
~~~~~~~~~~~~
The `perception` module depends on the Berkeley AutoLab's `core`_ module,
which can be installed using `pip install` on the source repo.
The module primarily wraps `OpenCV`_ version >= 2.11 which can be installed using pip.

Furthermore, the `perception` module optionally depends on `Tensorflow`_ and `pylibfreenect2`_ for Convolutional Neural Networks and Kinect2 sensor usage, repectively.
Install these according to their website's instructions if their functionality is required.

However, installing our repo using `pip` will install these automatically.

.. _core: https://github.com/mmatl/core
.. _OpenCV: https://pypi.python.org/pypi/opencv-python
.. _Tensorflow: http://tflearn.org/installation/
.. _pylibfreenect2: http://r9y9.github.io/pylibfreenect2/installation.html

Any other dependencies will be installed automatically when `perception` is
installed with `pip`.

Cloning the Repository
~~~~~~~~~~~~~~~~~~~~~~
You can clone or download our source code from `Github`_. ::

    $ git clone git@github.com:mmatl/perception.git

.. _Github: https://github.com/mmatl/perception

Installation
~~~~~~~~~~~~
To install `perception` in your current Python environment, simply
change directories into the `perception` repository and run ::

    $ pip install -e .

or ::

    $ pip install -r requirements.txt

Alternatively, you can run ::

    $ pip install /path/to/perception

to install `perception` from anywhere.

Testing
~~~~~~~
To test your installation, run ::

    $ python setup.py test

We highly recommend testing before using the module.

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~
Building `perception`'s documentation requires a few extra dependencies --
specifically, `sphinx`_ and a few plugins.

.. _sphinx: http://www.sphinx-doc.org/en/1.4.8/

To install the dependencies required, simply run ::

    $ pip install -r docs_requirements.txt

Then, go to the `docs` directory and run `make` with the appropriate target.
For example, ::

    $ cd docs/
    $ make html

will generate a set of web pages. Any documentation files
generated in this manner can be found in `docs/build`.

