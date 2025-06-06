"""
    FourMomentum{T<:Real}(en::T, x::T, y::T, z::T)

Defines a four-momentum vector energy and cartesian spatial components, where the type `T` 
is a subtype of `Real`, allowing for arbitrary precision and numeric types.

# Fields

- `en`: Energy component.
- `x`: Spatial component in the x-direction.
- `y`: Spatial component in the y-direction.
- `z`: Spatial component in the z-direction.

# Example

```jldoctest
julia> FourMomentum(4.0, 1.0, 2.0, 3.0)
FourMomentum(en = 4.0, x = 1.0, y = 2.0, z = 3.0)

julia> FourMomentum(4, 1.0, 2, 3) # implicit type promotion
FourMomentum(en = 4.0, x = 1.0, y = 2.0, z = 3.0)
```
"""
struct FourMomentum{T<:Real} 
    en::T  # Energy component
    x::T   # Spatial component in the x-direction
    y::T   # Spatial component in the y-direction
    z::T   # Spatial component in the z-direction
end
# type promotion on construction
FourMomentum(en,x,y,z) = FourMomentum(promote(en,x,y,z)...)

# return the element type
function Base.eltype(::FourMomentum{T}) where T 
    return T
end

# Overload Base.show for pretty printing of FourMomentum; plain text version
function Base.show(io::IO, m::MIME"text/plain", p::FourMomentum)
    println(io, """FourMomentum(en = $(p.en), x = $(p.x), y = $(p.y), z = $(p.z))""")
    return nothing
end

# Overload Base.show for pretty printing of FourMomentum; inline version
function Base.show(io::IO, p::FourMomentum)
    println(io, "($(round(p.en,digits=6)), $(round(p.x,digits=6)), $(round(p.y,digits=6)), $(round(p.z,digits=6)))")
    return nothing
end

"""

    Base.:+(p1::FourMomentum,p2::FourMomentum) 

Defines vector addition for two `FourMomentum` objects. The result is a new `FourMomentum` 
with each component being the sum of the corresponding components of `p1` and `p2`.

# Example

```julia
julia> p1 = FourMomentum(4.0, 1.0, 2.0, 3.0)
FourMomentum(en = 4.0, x = 1.0, y = 2.0, z = 3.0)


julia> p2 = FourMomentum(2.0, 0.5, 1.0, 1.5)
FourMomentum(en = 2.0, x = 0.5, y = 1.0, z = 1.5)


julia> p1 + p2
FourMomentum(en = 6.0, x = 1.5, y = 3.0, z = 4.5)
```
"""
function Base.:+(p1::FourMomentum, p2::FourMomentum) 

    #element-wise addition of the four-momentum components
    return FourMomentum(p1.en + p2.en, p1.x + p2.x, p1.y + p2.y, p1.z + p2.z)

end


"""
    
    Base.:-(p1::FourMomentum,p2::FourMomentum) 

Defines vector subtraction for two `FourMomentum` objects. The result is a new `FourMomentum` 
with each component being the difference between the corresponding components of `p1` and `p2`.

# Example

```jldoctest
julia> p1 = FourMomentum(4.0, 1.0, 2.0, 3.0)
FourMomentum(en = 4.0, x = 1.0, y = 2.0, z = 3.0)

julia> p2 = FourMomentum(2.0, 0.5, 1.0, 1.5)
FourMomentum(en = 2.0, x = 0.5, y = 1.0, z = 1.5)

julia> p1 - p2
FourMomentum(en = 2.0, x = 0.5, y = 1.0, z = 1.5)
```
""" 
function Base.:-(p1::FourMomentum, p2::FourMomentum) 
    # element-wise subtraction of the four-momentum components
    return FourMomentum(p1.en - p2.en, p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)
end


"""
    
    Base.:*(a::Real,p2::FourMomentum) 

Defines scalar multiplication for a `FourMomentum` object, scaling each component of the 
four-momentum by a scalar `a`.

# Example

```jldoctest
julia> p = FourMomentum(4.0, 1.0, 2.0, 3.0)
FourMomentum(en = 4.0, x = 1.0, y = 2.0, z = 3.0)

julia> 2 * p
FourMomentum(en = 8.0, x = 2.0, y = 4.0, z = 6.0)
```
"""
function Base.:*(a::Real, p::FourMomentum) 
    # scalar multiplication of the four-momentum components
    return FourMomentum(a * p.en, a * p.x, a * p.y, a * p.z)
end


"""

    minkowski_dot(p1::FourMomentum, p2::FourMomentum)

Computes the Minkowski dot product of two four-momentum vectors. The dot product uses the 
Minkowski metric `(+,-,-,-)`. For ``p_i = (E_i,p_i^x,p_i^y,p_i^z)`` with ``i=1,2``, the result is:

```math
    p_1 \\cdot p_2 = E_1E_2 - p_1^xp_2^x - p_1^yp_2^y + p_1^zp_2^z
```

# Example
```jldoctest
julia> p1 = FourMomentum(4.0, 1.0, 2.0, 3.0)
FourMomentum(en = 4.0, x = 1.0, y = 2.0, z = 3.0)

julia> p2 = FourMomentum(3.0, 0.5, 1.0, 1.5)
FourMomentum(en = 3.0, x = 0.5, y = 1.0, z = 1.5)

julia> minkowski_dot(p1, p2)
5.0
```
"""
function minkowski_dot(p1::FourMomentum, p2::FourMomentum)
    # Minkowski metric: (+,-,-,-)
    return p1.en * p2.en - p1.x * p2.x - p1.y * p2.y - p1.z * p2.z
end

function _construct_moms_from_coords(E_in, cos_theta, phi)
    # Calculate the energy and momentum components for the electron-positron pair
    E = E_in / 2  # Each particle has half the total energy in the center-of-mass frame
    p = sqrt(E^2 - ELECTRON_MASS)  # Assuming electron mass is 0.511 MeV/c^2

    # Calculate the spatial components based on the scattering angle and azimuthal angle
    px = p * sqrt(1 - cos_theta^2) * cos(phi)
    py = p * sqrt(1 - cos_theta^2) * sin(phi)
    pz = p * cos_theta

    # Construct four-momenta for e-, e+, mu-, mu+
    e_minus = FourMomentum(E, px, py, pz)
    e_plus = FourMomentum(E, -px, -py, -pz)
    mu_minus = FourMomentum(E, px, py, pz)
    mu_plus = FourMomentum(E, -px, -py, -pz)

    return Dict("e-" => e_minus, "e+" => e_plus, "mu-" => mu_minus, "mu+" => mu_plus)
end

# TODO: 
# consider using NamedTuples instead





"""
    coords_to_dict(E_in::Real, cos_theta::Real, phi::Real)

Constructs the four-momenta for an electron-positron annihilation process ``e^+ e^- \\rightarrow \\mu^+ \\mu^-``
in the center-of-mass frame. The input energy (`E_in`), scattering angle (`cos_theta`), and azimuthal angle (`phi`) 
are used to compute the incoming and outgoing momenta for the particles.

# Returns
A `Dict` mapping the particle names ("e-", "e+", "mu-", "mu+") to their respective `FourMomentum` objects.

# Example

```jldoctest
julia> mom_dict = coords_to_dict(1e3,0.9,pi/4)
Dict{String, FourMomentum{Float64}} with 4 entries:
  "mu+" => (1000.0, -306.495431, -306.495431, -894.962239)…
  "mu-" => (1000.0, 306.495431, 306.495431, 894.962239)…
  "e+"  => (1000.0, 0.0, 0.0, -999.999869)…
  "e-"  => (1000.0, 0.0, 0.0, 999.999869)…

julia> mom_dict["e-"]
FourMomentum(en = 1000.0, x = 0.0, y = 0.0, z = 999.999869440028)


julia> mom_dict["e+"]
FourMomentum(en = 1000.0, x = 0.0, y = 0.0, z = -999.999869440028)


julia> mom_dict["mu-"]
FourMomentum(en = 1000.0, x = 306.4954310103767, y = 306.49543101037665, z = 894.9622389946002)


julia> mom_dict["mu+"]
FourMomentum(en = 1000.0, x = -306.4954310103767, y = -306.49543101037665, z = -894.9622389946002)
```
"""
function coords_to_dict(E_in,cos_theta,phi)

    # Validate input types
    if !(isa(E_in, Real) && isa(cos_theta, Real) && isa(phi, Real))
        throw(ArgumentError("All inputs must be of type Real."))
    end

    # Validate ranges
    if cos_theta < -1 || cos_theta > 1
        throw(ArgumentError("cos_theta must be in the range [-1, 1]."))
    end
    if phi < 0 || phi >= 2 * pi
        throw(ArgumentError("phi must be in the range [0, 2π)."))
    end

    return _construct_moms_from_coords(E_in, cos_theta, phi)


    
end

function extractThreeMomentum(p::FourMomentum)
    
    return (p.x, p.y, p.z)
end

function extractEnergy(p::FourMomentum)
    
    return p.en
end

function extractNegativeThreeMomentum(p::FourMomentum)
    
    return (-p.x, -p.y, -p.z)
end

