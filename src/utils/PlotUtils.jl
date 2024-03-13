"""
    waterfall_plot(x, y, z; zmin = minimum(z), lw = 1., colorline = :auto, colmap = :viridis, colorband = (:white, 1.), xlab = "x", ylab = "y", zlab = "z", edge = true, axis_tight = false)

Plot a waterfall plot.

# Inputs
- `x::Vector`: x-axis values
- `y::Vector`: y-axis values
- `z::Matrix`: z-axis values
- `zmin::Real`: minimum value of z-axis
- `lw::Real`: linewidth
- `colorline::Symbol`: color of the lines
- `colmap::Symbol`: colormap
- `colorband::Tuple`: color of the band
- `xlab::String`: x-axis label
- `ylab::String`: y-axis label
- `zlab::String`: z-axis label
- `edge::Bool`: display edges
- `axis_tight::Bool`: tight axis

# Output
- `fig::Figure`: figure

# Example
```julia-repl
julia> waterfall(1:10, 1:10, rand(10, 10))
```
"""
function waterfall_plot(x, y, z; zmin = minimum(z), lw = 1., colorline = :auto, colmap = :viridis, colorband = (:white, 1.), xlab = "x", ylab = "y", zlab = "z", edge = true, axis_tight = false, xlim = [minimum(x), maximum(x)], ylim = [minimum(y), maximum(y)], zlim = [zmin, maximum(z)])
    # Initialisation
    ny = length(y)
    I₂ = ones(2)

    fig = Figure()
    ax = Axis3(fig[1,1], xlabel = xlab, ylabel = ylab, zlabel = zlab)
    for (j, yv) in enumerate(reverse(y))
        idz = ny - j + 1
        zj = z[idz, :]
        lower = Point3f.(x, yv, zmin)
        upper = Point3f.(x, yv, zj)
        band!(ax, lower, upper, color = colorband)

        if edge
            edge_start = [Point3f(x[1], yv, zmin), Point3f(x[1], yv, zj[1])]
            edge_end = [Point3f(x[end], yv, zmin), Point3f(x[end], yv, zj[end])]
        end

        if colorline == :auto
            lines!(ax, upper, color = zj, colormap = colmap, linewidth = lw)

            if edge
                lines!(ax, edge_start, color = zj[1]*I₂, colormap = colmap, linewidth = lw)
                lines!(ax, edge_end, color = zj[end]*I₂, colormap = colmap, linewidth = lw)
            end
        else
            lines!(ax, upper, color = colorline, linewidth = lw)

            if edge
                lines!(ax, edge_start, color = colorline, linewidth = lw)
                lines!(ax, edge_end, color = colorline, linewidth = lw)
            end
        end
    end

    if axis_tight
        xlims!(ax, minimum(x), maximum(x))
        ylims!(ax, minimum(y), 1.01*maximum(y))
        zlims!(ax, zmin, maximum(z))
    else
        xlims!(ax, xlim[1], xlim[2])
        ylims!(ax, ylim[1], ylim[2])
        zlims!(ax, zlim[1], zlim[2])
    end

    # Font
    labelsize = 18.
    ticklabelsize = 14.

    ax.xlabelsize = labelsize
    ax.xlabelfont = :bold

    ax.ylabelsize = labelsize
    ax.ylabelfont = :bold

    ax.zlabelsize = labelsize
    ax.zlabelfont = :bold
    ax.zticklabelpad = 5.

    ax.xticklabelsize = ticklabelsize
    ax.yticklabelsize = ticklabelsize
    ax.zticklabelsize = ticklabelsize

    return fig
end

function plot(x, y1, y2; xlab = "x", ylab = "y", lw = 1., col = (:blue, :red), ls = (:solid, :dash))
    fig = Figure()
    ax = Axis(fig[1,1], xlabel = xlab, ylabel = ylab)
    lines!(ax, x, y1, color = col[1], linewidth = lw, linestyle = ls[1], label = "Reference")
    lines!(ax, x, y2, color = col[2], linewidth = lw, linestyle = ls[2], label = "Estimation")

    xlims!(ax, minimum(x), maximum(x))
    axislegend(position = :rb, backgroundcolor = (:white, 0.75))
    # Font
    labelsize = 18.
    ticklabelsize = 14.

    ax.xlabelsize = labelsize
    ax.xlabelfont = :bold

    ax.ylabelsize = labelsize
    ax.ylabelfont = :bold

    ax.xticklabelsize = ticklabelsize
    ax.yticklabelsize = ticklabelsize

    return fig
end

function plot(x, y1, y2, y3; xlab = "x", ylab = "y", lw = 1.)
    fig = Figure()
    ax = Axis(fig[1,1], xlabel = xlab, ylabel = ylab)
    lines!(ax, x, y2, color = :red, linewidth = lw, linestyle = :dash)
    lines!(ax, x, y1, color = :blue, linewidth = lw)
    lines!(ax, x, y3, color = :black)

    xlims!(ax, minimum(x), maximum(x))

    # Font
    labelsize = 18.
    ticklabelsize = 14.

    ax.xlabelsize = labelsize
    ax.xlabelfont = :bold

    ax.ylabelsize = labelsize
    ax.ylabelfont = :bold

    ax.xticklabelsize = ticklabelsize
    ax.yticklabelsize = ticklabelsize

    return fig
end