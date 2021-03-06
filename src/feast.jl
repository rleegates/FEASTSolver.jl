import LinearAlgebra: qr, lu, eigen, diag

function feast!(X::AbstractMatrix, A::AbstractMatrix;
				nodes::Integer=8, iter::Integer=10, c=complex(0.0,0.0), r=1.0, ϵ=1e-12,
				debug=false, store=false, mixed_prec=false, factorizer=lu, left_divider=ldiv!)
	contour = circular_contour_trapezoidal(c, r, nodes)
	feast!(X, A, contour; iter=iter, debug=debug, ϵ=ϵ, store=store, mixed_prec=mixed_prec, factorizer=factorizer, left_divider=left_divider)
end

finalize!(x::Any) = nothing


function feast!(X::AbstractMatrix, A::AbstractMatrix, contour::Contour;
					 iter::Integer=10, ϵ=1e-12, debug=false, store=false, mixed_prec=false, factorizer=lu, left_divider=ldiv!)
	N, m₀ = size(X)
	if size(A, 1) != size(A, 2)
		 error("Incorrect dimensions of A, must be square")
	elseif size(A,1) != N
		 error("Incorrect dimensions of X, must match A")
	end

	Ctype = if mixed_prec ComplexF32 else ComplexF64 end

	Λ, resolvent, res = zeros(ComplexF64, m₀), zeros(ComplexF64, m₀), zeros(m₀)
	temp, R, Q = zeros(Ctype, N, m₀), similar(X, ComplexF64), deepcopy(X)
	Aq, Xq = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)

	ZmA = similar(A, Ctype)
	nodes = size(contour.nodes, 1)

	if store
		ZmA .= (A - I*contour.nodes[1])
		facts1 = factorizer(ZmA)
		facts = Array{typeof(facts1)}(undef, nodes)
		facts[1] = facts1

		Threads.@threads for i=2:nodes
			  ZmA .= (A - I*contour.nodes[i])
			  facts[i] = factorizer(ZmA)
		end
	end

	for nit=0:iter
		Q .= Matrix(qr(Q).Q)
		mul!(R, A, Q) ## why does this one allocate?
		mul!(Aq, Q', R) ### Aq = Q' * A * Q
		# mul!(Bq, Q', Q) ### Bq = Q' * Q = I
		F = eigen!(Aq)
		Λ .= F.values
		Xq .= F.vectors
		mul!(X, Q, Xq) ### Recover eigenvectors from Ritz vectors ( X = Q * Xq )
		update_R!(X, R, Λ, A) ### compute residual vectors R for RII update
		residuals!(res, R, Λ, A) ### compute actual residuals
		contour_nonempty = reduce(|, in_contour(Λ, contour))
		if debug iter_debug_print(nit, Λ, res, contour, 1e-5) end
		if contour_nonempty && maximum(res[in_contour(Λ, contour)]) < ϵ
			if debug println("converged in $nit iteration") end
			break
		end
		if nit < iter ### Do not solve linear systems / form Q on last iteration
			Q .= 0.00
			for i=1:nodes
				resolvent .= 1.0 ./(contour.nodes[i] .- Λ)
				if store
					left_divider(temp, facts[i], R)
				else
					ZmA .= A - I*contour.nodes[i]
					linsolve!(temp, ZmA, R, factorizer, left_divider)
				end

				temp .= X - temp
				rmul!(temp, Diagonal(resolvent .* contour.weights[i]))
				Q .+= temp
			end
		end
	end
	if store
		foreach(finalize!, facts)
	end
	contour_nonempty = reduce(|, in_contour(Λ, contour))
	if !contour_nonempty println("no eigenvalues found in contour!") end
	Λ[in_contour(Λ, contour)], X[:,in_contour(Λ, contour)], res[in_contour(Λ, contour)]
end

function gen_feast!(X::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix;
					nodes::Integer=8, iter::Integer=10, c=complex(0.0,0.0), r=1.0,
					debug=false, store=false, ϵ=1e-12, factorizer=lu, left_divider=ldiv!)
	contour = circular_contour_trapezoidal(c, r, nodes)
	gen_feast!(X, A, B, contour; iter=iter, debug=debug, ϵ=ϵ, factorizer=factorizer, left_divider=ldiv!)
end

function gen_feast!(X::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, contour::Contour;
					iter::Integer=10, debug=false, store=false, ϵ=1e-12, factorizer=lu, left_divider=ldiv!)
	N, m₀ = size(X)
	if size(A, 1) != size(A, 2)
		error("Incorrect dimensions of A, must be square")
	elseif size(A,1) != N
		error("Incorrect dimensions of X, must match A")
	end

	Λ, resolvent, res = zeros(ComplexF64, m₀), zeros(ComplexF64, m₀), zeros(m₀)
	temp, R, Q = zeros(ComplexF64, N, m₀), similar(X, ComplexF64), copy(X)
	Aq, Bq, Xq = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)
	ZmA = similar(A, ComplexF64)
	nodes = size(contour.nodes, 1)

	if store
		 ZmA .= (A - B*contour.nodes[1])
		 facts1 = factorizer(ZmA)
		 facts = Array{typeof(facts1)}(undef, nodes)
		 facts[1] = facts1

		 Threads.@threads for i=2:nodes
				ZmA .= (A - B*contour.nodes[i])
				facts[i] = factorizer(ZmA)
		 end
	end

	for nit=0:iter
		Q .= Matrix(qr(Q).Q)
		mul!(R, A, Q) ## why does this one allocate?
		mul!(Aq, Q', R) ### Aq = Q' * A * Q
		mul!(R, B, Q)
		mul!(Bq, Q', R) ### Bq = Q' * Q = I
		F = eigen!(Aq, Bq)
		Λ .= F.values
		Xq .= F.vectors
		mul!(X, Q, Xq) ### Recover eigenvectors from Ritz vectors ( X = Q * Xq )
		update_R!(X, R, Λ, A, B) ### compute residual vectors R for RII update
		residuals!(res, R, Λ, A) ### compute actual residuals
		contour_nonempty = reduce(|, in_contour(Λ, contour))
		if debug iter_debug_print(nit, Λ, res, contour, 1e-5) end
		if contour_nonempty && maximum(res[in_contour(Λ, contour)]) < ϵ
			  if debug println("converged in $nit iteration") end
			  break
		end
		if nit < iter ### Do not solve linear systems / form Q on last iteration
			Q .= 0.00
			for i=1:nodes
				resolvent .= (1.0 ./(contour.nodes[i] .- Λ))
				if store
					left_divider(temp, facts[i], R)
				else
					ZmA .= (A - B*contour.nodes[i])
					linsolve!(temp, ZmA, R, factorizer, left_divider)
				end
				temp .= X - temp
				rmul!(temp, Diagonal(resolvent .* contour.weights[i]))
				Q .+= temp
			end
		end
	end
	if store
		foreach(finalize!, facts)
	end
	contour_nonempty = reduce(|, in_contour(Λ, contour))
	if !contour_nonempty println("no eigenvalues found in contour!") end
	Λ[in_contour(Λ, contour)], X[:,in_contour(Λ, contour)], res[in_contour(Λ, contour)]
end

function dual_gen_feast!(Xr::AbstractMatrix, Xl::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix;
					nodes::Integer=8, iter::Integer=10, c=complex(0.0,0.0), r=1.0,
					debug=false, store=false, ϵ=1e-12, factorizer=lu, left_divider=ldiv!)
	contour = circular_contour_trapezoidal(c, r, nodes)
	dual_gen_feast!(Xr, Xl, A, B, contour; iter=iter, debug=debug, ϵ=ϵ, factorizer=factorizer, left_divider=ldiv!)
end

function dual_gen_feast!(Xr::AbstractMatrix, Xl::AbstractMatrix, A::AbstractMatrix, B, contour::Contour;
					iter::Integer=10, debug=false, store=false, ϵ=1e-12, factorizer=lu, left_divider=ldiv!, resize_subspace=true)
	N, m₀ = size(Xl)
	if size(A, 1) != size(A, 2)
		error("Incorrect dimensions of A, must be square")
	elseif size(A,1) != N
		error("Incorrect dimensions of X, must match A")
	end

	Λ, resolvent, resr, resl, temp, Rr, Rl, Ql, Qr, Aq, Bq, Xq = init(Xr, Xl, N, m₀)
	ZmA = similar(A, ComplexF64)
	nodes = size(contour.nodes, 1)


	if store
		if debug print("Precomputing factorizations: ") end
			ZmA .= (A - B*contour.nodes[1])
			rfacts1 = factorizer(ZmA)
			rfacts = Array{typeof(rfacts1)}(undef, nodes)
			rfacts[1] = rfacts1
			if debug print("1R ") end
			ZmA .= (A - B*contour.nodes[1])'
			lfacts1 = factorizer(ZmA)
			lfacts = Array{typeof(lfacts1)}(undef, nodes)
			lfacts[1] = lfacts1
			if debug print("1L ") end
			for i=2:nodes
				ZmA .= (A - B*contour.nodes[i])
				rfacts[i] = factorizer(ZmA)
				if debug print("$(i)R ") end
				ZmA .= (A - B*contour.nodes[i])'
				lfacts[i] = factorizer(ZmA)
				if debug print("$(i)L ") end
			end
		if debug println() end
	end

	try
	nin = 0
	for nit=0:iter
		if resize_subspace
			new_m₀ = new_subspace_size(nin, m₀)
			if new_m₀ > 0 && new_m₀ != m₀
				if debug println("Resizing subspace from $(m₀) to $(new_m₀)") end
				m₀ = new_m₀
				Λ, Xr, Xl, resolvent, resr, resl, temp, Rr, Rl, Ql, Qr, Aq, Bq, Xq = reinit(Λ, Xr, Xl, resolvent, resr, resl, temp, Rr, Rl, Ql, Qr, Aq, Bq, Xq, new_m₀)
			end
		end
		S = svd!(Ql'*B*Qr)
		Qr .= Qr*S.V*Diagonal(1.0/S.S)
		Ql .= Ql*S.U*Diagonal(1.0/S.S)
		mul!(Rr, A, Qr) ## why does this one allocate?
		mul!(Aq, Ql', Rr) ### Aq = Q' * A * Q
		mul!(Rr, B, Qr)
		mul!(Bq, Ql', Rr) ### Bq = Q' * Q = I
		F = eigen(Aq, Bq)
		Λ .= F.values
		Xq .= F.vectors
		mul!(Xr, Qr, Xq) ### Recover eigenvectors from Ritz vectors ( X = Q * Xq )
		F = eigen(Aq', Bq')
		Xq .= F.vectors
		mul!(Xl, Ql, Xq) ### Recover eigenvectors from Ritz vectors ( X = Q * Xq )
		update_R!(Xr, Rr, Λ, A, B) ### compute residual vectors R for RII update
		update_R!(Xl, Rl, Λ, A', B') ### compute residual vectors R for RII update
		residuals!(resr, Rr, Λ, A) ### compute actual residuals
		incntr = in_contour(Λ, contour)
		nin = sum(resr[incntr] .< 1e-5)
		contour_nonempty = reduce(|, incntr)
		if debug iter_debug_print(nit, Λ, resr, contour, 1e-5) end
		if contour_nonempty && maximum(resr[in_contour(Λ, contour)]) < ϵ
			if debug println("converged in $nit iteration") end
			break
		end
		if nit < iter ### Do not solve linear systems / form Q on last iteration
			Qr .= 0.00
			Ql .= 0.00
			for i=1:nodes
				resolvent .= (1.0 ./(contour.nodes[i] .- Λ))
				if store
					left_divider(temp, rfacts[i], Rr)
				else
					ZmA .= (A - B*contour.nodes[i])
					linsolve!(temp, ZmA, Rr, factorizer, left_divider)
				end
				temp .= Xr - temp
				rmul!(temp, Diagonal(resolvent .* contour.weights[i]))
				Qr .+= temp

				resolvent .= (1.0 ./(conj(contour.nodes[i]) .- conj.(Λ)))
				if store
					left_divider(temp, lfacts[i], Rl)
				else
					ZmA .= (A - B*contour.nodes[i])'
					linsolve!(temp, ZmA, Rl, factorizer, left_divider)
				end
				temp .= Xl - temp
				rmul!(temp, Diagonal(resolvent .* conj(contour.weights[i])))
				Ql .+= temp
			end
		end
	end

	contour_nonempty = reduce(|, in_contour(Λ, contour))
	if !contour_nonempty
		println("no eigenvalues found in contour!")
	end
	output = Λ[in_contour(Λ, contour)], Xr[:,in_contour(Λ, contour)], Xl[:,in_contour(Λ, contour)], resr[in_contour(Λ, contour)]
	return output
	finally
		if store
			foreach(finalize!, rfacts)
			foreach(finalize!, lfacts)
		end
	end
end

function linsolve!(Y, C, X, factorizer, left_divider)
	F = factorizer(C)
	left_divider(Y, F, X)
	finalize!(F)
end



function init(Xr, Xl, N::Int, m₀::Int)
	Λ, resolvent, resr, resl = zeros(ComplexF64, m₀), zeros(ComplexF64, m₀), zeros(m₀), zeros(m₀)
	temp, Rr, Rl, Ql, Qr = zeros(ComplexF64, N, m₀), similar(Xr, ComplexF64), similar(Xr, ComplexF64), copy(Xl), copy(Xr)
	Aq, Bq, Xq = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)
	return Λ, resolvent, resr, resl, temp, Rr, Rl, Ql, Qr, Aq, Bq, Xq
end

function reinit(Λ, Xr, Xl, resolvent, resr, resl, temp, Rr, Rl, Ql, Qr, Aq, Bq, Xq, new_m₀, fillf::F=x->randn(typeof(x))) where F
	N, m₀ = size(temp,1), length(Λ)
	Xrn, Xln = zeros(ComplexF64, N, new_m₀), zeros(ComplexF64, N, new_m₀)
	map!(fillf, Xrn, Xrn)
	map!(fillf, Xln, Xln)
	copyto!(Xrn, Xr)
	copyto!(Xln, Xl)
	Λn, resolventn, resrn, resln, tempn, Rrn, Rln, Qln, Qrn, Aqn, Bqn, Xqn = init(Xrn, Xln, N, new_m₀)
	copyto!(Λn, Λ)
	copyto!(resolventn, resolvent)
	copyto!(resrn, resr)
	copyto!(resln, resl)
	copyto!(tempn, temp)
	copyto!(Rrn, Rr)
	copyto!(Rln, Rl)
	copyto!(Qln, Ql)
	copyto!(Qrn, Qr)
	copyto!(Aqn, Aq)
	copyto!(Bqn, Bq)
	copyto!(Xqn, Xqn)
	return Λn, Xrn, Xln, resolventn, resrn, resln, tempn, Rrn, Rln, Qln, Qrn, Aqn, Bqn, Xqn
end

function new_subspace_size(nin, m₀, fac_test=0.75, fac_new=2.0)
	if nin > ceil(Int, fac_test*m₀)
		return ceil(Int, fac_new*nin)
	else
		return 0
	end
end
