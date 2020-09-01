

function contour_estimate_eig(A::AbstractMatrix, contour::Contour, B=I;
                samples::Integer=min(100, size(A,1)), ϵ=1e-12, debug=false,
                mixed_prec=false, factorizer=lu, left_divider=ldiv!)


    N, m₀ = size(A,1), samples
    Ctype = if mixed_prec ComplexF32 else ComplexF64 end
    # X = if typeof(A) <: AbstractSparseMatrix
    #     sprandn(Ctype, N, m₀, min(1.0, m₀^2/N))
    # else
    #     randn(Ctype, N, m₀)
    # end
    ### Using sparse X does not work, possibly due to not being i.i.d. mean zero RV
    X = randn(Ctype, N, m₀)

    temp = zeros(Ctype, N, m₀)
    P = zeros(ComplexF64, N, m₀)
    ZmA = similar(A, Ctype)
    nodes = size(contour.nodes, 1)
    est = 0.0

    for i=1:nodes
        ZmA .= Ctype.(B*contour.nodes[i] - A)
		F = factorizer(ZmA)
		est += estimate_trace!(P, temp, X, F, B, left_divider=left_divider) * contour.weights[i]
        if debug print(".") end
    end
    if debug println() end

    return real(est)
end

# function contour_estimate_eig(A::AbstractMatrix, contour::Contour, B=I;
#                 samples::Integer=min(100, size(A,1)), ϵ=1e-12, debug=false,
#                 mixed_prec=false, factorizer=lu, left_divider=ldiv!)
#
#
#     N, m₀ = size(A,1), samples
#     Ctype = if mixed_prec ComplexF32 else ComplexF64 end
#     # X = if typeof(A) <: AbstractSparseMatrix
#     #     sprandn(Ctype, N, m₀, min(1.0, m₀^2/N))
#     # else
#     #     randn(Ctype, N, m₀)
#     # end
#     ### Using sparse X does not work, possibly due to not being i.i.d. mean zero RV
#     X = randn(Ctype, N, m₀)
#
#     temp = zeros(Ctype, N, m₀)
#     P = zeros(ComplexF64, m₀, m₀)
#     ZmA = similar(A, Ctype)
#     nodes = size(contour.nodes, 1)
#     est = 0.0
#
#     for i=1:nodes
#         ZmA .= Ctype.(B*contour.nodes[i] - A)
#         linsolve!(temp, ZmA, X, factorizer, left_divider)
#         mul!(P, X', temp)
#         est += tr(P)*contour.weights[i]/samples
#         if debug print(".") end
#     end
#     if debug println() end
#
#     return real(est)
# end


estimate_trace(A, N) = estimate_trace(convert(typeof(A), sparse(I, size(A)...)), A, N)

estimate_trace(A::AbstractMatrix, B::AbstractMatrix, N) = begin
	@assert size(A,1) == size(A,2)
	@assert size(B,1) == size(B,2)
	M = size(A,1)
	T = promote_type(eltype(A), eltype(B))
	y = zeros(T,M,N)
	x, b = copy(y), copy(y)
	res = estimate_trace!(b, x, y, lu(A), B)
	return res
end

function estimate_trace!(b::AbstractMatrix, x::AbstractMatrix, y::AbstractMatrix, Af::F, B::AbstractMatrix; left_divider::F1=ldiv!, randfill::Bool=true) where {F, F1}
	# Estimate trace of inv(A)*B where Af is a factorization of type F
	@assert size(B,1) == size(B,2)
	M = size(B,2)
	@assert size(y,1) == M
	@assert size(x) == size(y) == size(b)
	N = size(y,2)
	T = eltype(y)
	# Fill sample vectors with iid random numbers
	if randfill @inbounds for i = 1:length(y) y[i] = randn(T) end end
	# Compute b = B*y
	mul!(b, B, y)
	# Compute x = inv(A)*B*y
	left_divider(x, Af, b)
	res = zero(T)
	# Sum over sample vectors
	@inbounds for j = 1:N
		# Compute the dot-product
		@simd for i = 1:M
			res += adjoint(y[i,j]) * x[i,j]
		end
	end
	return res / N
end
