Brightness maps
=====================================

Zhang and Showman 2017
-----------------------

main parameters:

	**xi**
		Ratio of radiative to advective timescale (unit: Unitless)

	**T_n**
		Temperature of the nightside of the planet (unit: Kelvin)

	**delta_T**
		Day-night temperature contrast (unit: Kelvin)

An example square plot:

.. figure:: images/zhang_t_square.png
    :width: 800px
    :align: center
    :alt: alternate text
    :figclass: align-center

An example four phase plot:

.. figure:: images/zhang_t_temp_map.png
    :width: 800px
    :align: center
    :alt: alternate text
    :figclass: align-center

The resulting lightcurves for several parameter values

.. figure:: images/zhang_change_eta.png
    :width: 800px
    :align: center
    :alt: alternate text
    :figclass: align-center


Offset hotspot
--------------------

main parameters:

	**la0**
		Offset of the center of the hotspot in the latitude direction (unit: Degrees)

	**lo0**
		Offset of the center of the hotspot in the longitude direction (unit: Degrees)

	**size**
		The radius of the hotspot in degrees, i.e., 90 means the hotspot covers a whole hemisphere. (unit: degrees)

The hotspot can either be specified as "hotspot_b", to directly specify the fractional brightness, in which case these parameters are used:

	**spot_b**
		The surface brightness of the hotspot as a fraction of the surface brightness of the star, typically of order ~1e-4 for hot Jupiters (unitless)

	**p_b**
		The surface brightness of the planet that is not in the hotspot as a fraction of the surface brightness of the star. This value will depend strongly on the physics of heat transport in the planets atmosphere and may be several orders of magnitude fainter than the spot (unitless)

Or as "hotspot_t" to specify in terms of brightness temperature, in which case the following parameters are used instead. In this case the wavelength range to integrate over must be specified.

	**spot_T**
		The surface brightness of the hotspot as a fraction of the surface brightness of the star, typically of order ~1e-4 for hot Jupiters (unitless)

	**p_T**
		The brightness temperature of the planet that is not in the hotspot as a fraction of the surface brightness of the star. This value will depend strongly on the physics of heat transport in the planets atmosphere and may be several orders of magnitude fainter than the spot (unitless)

.. note::  Because there is a sharp contrast in flux levels between *spot* and *not spot* regions, this brightness model can have issues with quantisation, which produces unphysical "steps" in the lightcurve. This can be for the time being be solved by including a numerical integration step in regions with sharp contrasts with the optional paramter "grid_size"

cont

	**grid_size**
		This model has a sharp boundary, so can have quantization issues. Regions with sharp changes in brightness are for now integrated numerically instead of analytically, this sets the number of grid points to use in the integration along each direction, to the total number of additional function calls will be this value squared. Setting this too high can significantly slow the code down, however if it is too low fits may be numerically unstable. Use caution. This is a temporary fix and is intended to be removed in a future version (default: 10)


An example square plot:

.. figure:: images/hotspot_t_square.png
    :width: 800px
    :align: center
    :alt: alternate text
    :figclass: align-center

An example four phase plot:

.. figure:: images/hotspot_t_temp_map.png
    :width: 800px
    :align: center
    :alt: alternate text
    :figclass: align-center

The resulting lightcurves for several parameter values

.. figure:: images/hotspot_t_change_offset.png
    :width: 800px
    :align: center
    :alt: alternate text
    :figclass: align-center

Two sided planet
--------------------

This is a simple model that only assumes that the day-side of the planet has a different flux or temperature to the night side. can be called as "two temperature" to specify with brightness temperature, or "two brightness" to secify by flux

main parameters:

The hotspot can either be specified as "hotspot_b", to directly specify the fractional brightness, in which case these parameters are used:

	**pb_d**
		The surface brightness of the dayside as a fraction of the surface brightness of the star, typically of order ~1e-4 for hot Jupiters (unitless)

	**pb_n**
		The surface brightness of the planet nightside as a fraction of the surface brightness of the star. This value will depend strongly on the physics of heat transport in the planets atmosphere and may be several orders of magnitude fainter than the spot (unitless)

Or as "hotspot_t" to specify in terms of brightness temperature, in which case the following parameters are used instead. In this case the wavelength range to integrate over must be specified.

	**spot_T**
		The surface brightness of the hotspot as a fraction of the surface brightness of the star, typically of order ~1000 K for hot Jupiters (unit: kelvin)

	**p_T**
		The brightness temperature of the planet that is not in the hotspot. This value will depend strongly on the physics of heat transport in the planets atmosphere and may be significantly cooler than the spot (unit: degrees)

.. note::  Because there is a sharp contrast in flux levels between *spot* and *not spot* regions, this brightness model can have issues with quantisation, which produces unphysical "steps" in the lightcurve. This can be for the time being be solved by including a numerical integration step in regions with sharp contrasts with the optional paramter "grid_size"

cont

	**grid_size**
		This model has a sharp boundary, so can have quantization issues. Regions with sharp changes in brightness are for now integrated numerically instead of analytically, this sets the number of grid points to use in the integration along each direction, to the total number of additional function calls will be this value squared. Setting this too high can significantly slow the code down, however if it is too low fits may be numerically unstable. Use caution. This is a temporary fix and is intended to be removed in a future version (default: 10)


An example square plot:

.. figure:: images/daynight_t_square.png
    :width: 800px
    :align: center
    :alt: alternate text
    :figclass: align-center

An example four phase plot:

.. figure:: images/daynight_temp_map.png
    :width: 800px
    :align: center
    :alt: alternate text
    :figclass: align-center

The resulting lightcurves for several parameter values

.. figure:: images/daynight_change_night.png
    :width: 800px
    :align: center
    :alt: alternate text
    :figclass: align-center