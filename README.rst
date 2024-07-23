+-----------------------+
| **Documentation**     |
+=======================+
| |Documentation|       |
+-----------------------+

.. |Documentation| image:: https://img.shields.io/badge/api-reference-blue.svg
   :target: https://arpes-v4.readthedocs.io/en/daredevil/

|coverage| |docs_status| |code_format| |code style| |rye| 


.. |docs_status| image:: https://readthedocs.org/projects/arpes-v4/badge/?version=stable&style=flat
   :target: https://arpes-v4.readthedocs.io/en/stable/
.. |coverage| image:: https://codecov.io/gh/arafune/arpes/graph/badge.svg?token=TW9EPVB1VE
   :target:  https://app.codecov.io/gh/arafune/arpes
.. |code style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |code_format| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
.. |rye| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/rye/main/artwork/badge.json
    :target: https://rye-up.com
    :alt: Rye

PyARPES corrected  (V4)
=======================

.. image:: docs/source/_static/video/intro-video.gif

========

PyARPES simplifies the analysis and collection of angle-resolved photoemission spectroscopy (ARPES) and emphasizes

* modern, best practices for data science
* support for a standard library of ARPES analysis tools mirroring those available in Igor Pro
* (interactive and extensible analysis tools)

It supports a variety of data formats from synchrotron and laser-ARPES sources including ARPES at the Advanced
Light Source (ALS), the data produced by Scienta Omicron GmbH's "SES Wrapper", data and experiment files from
Igor Pro, NeXuS files, and others.

To learn more about installing and using PyARPES in your analysis or data collection application,
visit `the documentation site`_.

PyARPES is currently developed by Conrad Stansbury of the Lanzara Group at the University of California, Berkeley.

Citing PyARPES
--------------

If you use PyARPES in your work, please support the development of scientific software by acknowledging its usefulness to you with a citation.
The simplest way to do this is to cite the paper describing the package in SoftwareX


    @article{
        stansburypyarpes,
        title = {PyARPES: An analysis framework for multimodal angle-resolved photoemission spectroscopies},
        journal = {SoftwareX},
        volume = {11},
        pages = {100472},
        year = {2020},
        issn = {2352-7110},
        doi = {https://doi.org/10.1016/j.softx.2020.100472},
        url = {https://www.sciencedirect.com/science/article/pii/S2352711019301633},
        author = {Conrad Stansbury and Alessandra Lanzara},
        keywords = {ARPES, NanoARPES, Pump-probe ARPES, Photoemission, Python, Qt, Jupyter},
        abstract = {},
    }


Installation
============

PyARPES (>= V.4.0) can be installed from source.   Python version 3.11 or newer is strongly recommmended.

The current version has been largely revised from the original version which are in PyPI and conda site.
Unfortunately, I don't have the right the upload the current version to these site, and I would not like to take over it from the original author.

The main purpose of revision of the package is to make this be reliable for us. Actually, the original version outputs the wrong results in many
case, especially for angle-momentum conversion.

Thus, the current package can be installed only through the github.


Pip installation
----------------

::

   pip install git+http://github.com/arafune/arpes


Local installation from source
------------------------------

If you want to modify the source for PyARPES as you use it, you might prefer a local installation from source.
Details can be found on `the documentation site`_.



Suggested steps
---------------

1. install `rye <https://rye-up.com>`__.
2. Clone or duplicate the folder structure in this repository by `git clone https://github.com/arafune/arpes.git`
3. `rye sync` in `arpes` directory
4. Activate `arpes` environment (the way to activate depends on the OS/shell: I have confirmed the arpes-V4 works on Mac/Linux/Windows).

Contact
=======

Very unfortunately, we cannot get any responses from the original author.  The comment below does not make sense at present.

Questions, difficulties, and suggestions can be directed to Conrad Stansbury (chstan@berkeley.edu)
or added to the repository as an issue. In the case of trouble, also check the `FAQ`_.



Copyright |copy| 2018-2019 by Conrad Stansbury, all rights reserved.

PyArpes contribution after `cadaaae`_, |copy| 2023-2024 by Ryuichi Arafune, all rights reserved.

.. |copy|   unicode:: U+000A9 .. COPYRIGHT SIGN


.. _cadaaae: https://github.com/arafune/arpes/commit/cadaaae0525d0889ef030cf18cf049da8fec2ee3
.. _Jupyter: https://jupyter.org/
.. _the documentation site: https://arpes-v4.readthedocs.io/en/daredevil
.. _contributing: https://arpes-v4.readthedocs.io/en/daredevil/contributing.html
.. _FAQ: https://arpes-v4.readthedocs.io/en/daredevil/faq.html

