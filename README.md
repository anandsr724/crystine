<div align="center">
  <img src="crystine/crysrine_logo_horizontal.png">
</div>

[![PyPI](https://badge.fury.io/py/crystine.svg)](https://badge.fury.io/py/crystine.svg)

Crystine is a data analysis library for computational chemistry , which currently supports VASP.

## Install

To install the current release

```
$ pip install crystine
```

To update Crystine to the latest version, add `--upgrade` flag to the above
command.

## Usage

Crystine supports command line arguements, once installed you can directly run one of the scrpits from the command line interface.

Currently, Crystine can do the following tasks :

* `crystine-ginfo`: Genrates essential info regading calculation such as band gap , VBM CBM position.
* `crystine-gmass`: Calculates effective mass from you band structure calculation.

---

### Options Available

##### `crystine-ginfo`:

```
--excel, 
generate an excel file with data of VBM CBM of all K-Points

Default: 1
```

##### `crystine-gmass`:

```
--excel, 
generate an excel file with with all the data extracted and used in calculation

Default: 1

--dat, 
use exisiting band.dat file , else the code generates its own .dat file

Default: 0

--shift_vb,
choose another vb , do this if you are getting inf / None in the mass

Default: 0

--shift_cb,
choose another cb, do this if you are getting inf / None in the mass

Default: 0

Note:
shift vb or cb are positive nunbers , absolute value of these numbers are added in cb and subtracted from vb.
  
```

#### *Example*

```shell
$ module load DL-Conda/3.9 #load your python environment
$ crystine-ginfo --excel 1
```
