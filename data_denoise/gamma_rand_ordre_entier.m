%% gf = gamma_rand_ordre_entier( L, mu, m, n )
%%
%% fonction de tirage gamma de parametres,
%% entier L = ordre
%% mu = moyenne
%% entiers m,n = dimensions de la matrice

function gf = gamma_rand_ordre_entier(L,mu,m,n)

p = m*n ;

g = mu * ( abs( ( randn( L, p ) + sqrt(-1) * randn( L, p ) ) / sqrt(2) ) ).^2 / L ;

if (L>1)
  sg = sum( g , 1 ) ;
else
  sg = g ;
end%if

gf = reshape( sg, m, n );

end%function
