\\ Paper three's Table 4 at 77 digits: the stopping mass under a constant cap, exact and smooth,
\\ and the constant of the optimal cost (src/causaldyn_bench/paper_three.py, table_four). Every
\\ check prints PASS or FAIL beside its residual or its value; the last line counts the failures.
\\
\\ A schedule 0 <= v_t <= cap costs F(v) = A sum_t v_t + K sum_t 1/(I0 + c S_(t-1)), S the prefix
\\ sum. Front-loading is optimal, so F depends on the total M alone. On round n's interval,
\\ (n - 1) cap <= M <= n cap, it is A M + K H_n + K (T - n)/(I0 + c M), with H_n the harmonic sum
\\ sum_(t < n) 1/(I0 + c cap t): convex, with a slope that jumps up at every round boundary. The
\\ EXACT mass S_d is its minimum. It lies on the first round n whose right slope at n cap is not
\\ negative, A (I0 + c n cap)^2 >= K c (T - n - 1), and there it is
\\ min(sqrt(K (T - n)/(A c)) - I0/c, n cap). The SMOOTH mass S_c reads the round as M/cap and
\\ solves A w^2 + (K/cap) w = K c T + K I0/cap, w = I0 + c M. The classical form
\\ S_0 = sqrt(K T/(A c)) - I0/c holds the horizon at T.
\\
\\ 1. the rule above against a check of every round's interval, a boundary optimum included;
\\ 2. two instances at T = 3 in closed form: cap = 100, where the relaxation misses, and cap = 4/5;
\\ 3. over a grid of plants, caps and horizons: S_d = S_c at the horizon T - rho_T with rho_T in
\\    [0, 1]; S_c - S_d = K rho_T/(A (w_c + w_d) + K/cap) < cap; and S_0 - S_d <= K/(2 A c cap)
\\    exactly when c cap rho_T <= I0 + (w_0 - w_d)/2;
\\ 4.-6. Table 4's panels (a), (b) and (d) against the digits the paper prints.
\\
\\ Run: timeout 300 gp -q -f scripts/paper_three_mass.gp </dev/null, then grep the output for FAIL
\\ and ***: gp exits 0 after an error, and skips the rest of the file.

\p 77
default(format, "g.12");
checks = 0;
fails = 0;
report(name, ok, shown) = checks++; if(!ok, fails++); print(if(ok, "PASS ", "FAIL "), name, ": ", shown);
zero(name, x) = report(name, abs(x) < 10^-60, x);
\\ a value against the digits a table prints: within half a unit of the last one
agree(name, x, printed, unit) = report(name, abs(x - eval(printed)) <= unit/2, Str(x, " prints as ", printed));

\\ the plant's knobs: b = 1, rr = 1/2, xt = 1, sigma = 7/10, eta = 3/5, I0 = 1
knob_b = 1; knob_rr = 1/2; knob_xt = 1; knob_sigma = 7/10; knob_eta = 3/5;
plant() = A = knob_b^2 + knob_rr; K = A*(knob_xt*(knob_rr - knob_b^2)/(knob_rr + knob_b^2)^2)^2; c = knob_eta/knob_sigma^2; I0 = 1;
plant();

\\ the first round whose right slope at its end is not negative; T - 1 always qualifies
rounds(T, kap) = {
  my(lo = 0, hi = T - 1, mid);
  while(lo < hi,
    mid = (lo + hi) \ 2;
    if(A*(I0 + c*mid*kap)^2 >= K*c*(T - mid - 1), hi = mid, lo = mid + 1));
  lo;
}
\\ [S_d, the round that delivers it]
exact(T, kap) = {
  my(n = rounds(T, kap));
  if(n == 0, return([0, 0]));
  [min(sqrt(K*(T - n)/(A*c)) - I0/c, n*kap), n];
}
harmonic_sum(n, kap) = (psi(n + I0/(c*kap)) - psi(I0/(c*kap)))/(c*kap);
optimum(T, kap) = {
  my(s = exact(T, kap));
  if(s[2] == 0, return(K*T/I0));
  A*s[1] + K*harmonic_sum(s[2], kap) + K*(T - s[2])/(I0 + c*s[1]);
}
smooth(tau, kap) = ((-K/kap + sqrt((K/kap)^2 + 4*A*(K*c*tau + K*I0/kap)))/(2*A) - I0)/c;
classical(T) = sqrt(K*T/(A*c)) - I0/c;
ceiling(kap) = K/(2*A*c*kap);
coefficient(kap) = (K + 4*A*I0*kap)*sqrt(K/(A*c))/(8*A*c*kap^2);
half_band() = sqrt(K/(A*c))/2;
\\ rho_T: the relaxation's quadratic solved for its horizon at the exact mass
rho(T, kap) = my(s = exact(T, kap)[1], w = I0 + c*s); T - s/kap - A*w^2/(K*c);
constant(kap) = -A*I0/c + K/(c*kap)*(log(sqrt(K/(A*c))/kap) - psi(I0/(c*kap)) - 1);
slope(kap) = (K*I0/(c*kap) - K/2 - K^2/(4*A*c*kap^2))*sqrt(A/(K*c));
\\ the optimum's cost less c_causal sqrt(T) + (K/(2 c cap)) ln T + C_cap
residual(T, kap) = optimum(T, kap) - 2*sqrt(A*K/c)*sqrt(T) - K/(2*c*kap)*log(T) - constant(kap);

\\ every round's interval, checked one by one: the clipped balance on each, the cheapest kept
candidates(T, kap) = {
  my(best = K*T/I0, mass = 0, h = 0, m, v);
  for(n = 1, T,
    h += 1/(I0 + c*(n - 1)*kap);
    m = max((n - 1)*kap, min(n*kap, sqrt(K*(T - n)/(A*c)) - I0/c));
    v = A*m + K*h + K*(T - n)/(I0 + c*m);
    if(v < best, best = v; mass = m));
  [mass, best];
}

print("1. the rule against every round's interval");
{
  foreach([[400, 3/100], [4000, 1/10], [4000, 1/1000], [10000, 1/100]], case,
    my(T = case[1], kap = case[2], s = exact(T, kap), all = candidates(T, kap));
    zero(Str("mass at T = ", T, ", cap = ", kap, ", ", if(s[1] == s[2]*kap, "on a boundary", "inside a round")), s[1] - all[1]);
    zero(Str("cost at T = ", T, ", cap = ", kap), optimum(T, kap) - all[2]));
}

print("\n2. T = 3 at A = 1, K = 4, c = 1, I0 = 1");
{
  A = 1; K = 4; c = 1; I0 = 1;
  my(s = exact(3, 100));
  zero("cap = 100: S_d = sqrt(8) - 1", s[1] - (sqrt(8) - 1));
  zero("cap = 100: F* = 3 + 4 sqrt(2)", optimum(3, 100) - (3 + 4*sqrt(2)));
  agree("cap = 100: the smooth mass", smooth(3, 100), "2.4499", 10^-4);
  zero("cap = 100: S_0 = sqrt(12) - 1", classical(3) - (sqrt(12) - 1));
  agree("cap = 100: S_0 - S_d", classical(3) - s[1], "0.6357", 10^-4);
  zero("cap = 100: against a ceiling of 1/50", ceiling(100) - 1/50);
  zero("cap = 100: S_c at T - rho_T is S_d", smooth(3 - rho(3, 100), 100) - s[1]);
  s = exact(3, 4/5);
  zero("cap = 4/5: S_d = 1", s[1] - 1);
  zero("cap = 4/5: F* = 83/9", optimum(3, 4/5) - 83/9);
  zero("cap = 4/5: rho_3 = 3/4", rho(3, 4/5) - 3/4);
  zero("cap = 4/5: S_c at T = 9/4 is 1", smooth(9/4, 4/5) - 1);
  agree("cap = 4/5: the smooth mass at T = 3", smooth(3, 4/5), "1.3218", 10^-4);
}

print("\n3. S_d = S_c(T - rho_T) and its bracket over a grid");
{
  my(plants = [[3/2, 2/27, 60/49, 1], [1, 4, 1, 1], [1, 3/10, 2, 1/2]],
     levels = [10^-3, 1/100, 3/100, 1/10, 10^(-1/2), 1/2, 4/5, 2, 10, 100],
     count = 0, boundary = 0, outside = 0, identity = 0, gap = 0, below = 0, over = 0, iff = 0);
  foreach(plants, p,
    A = p[1]; K = p[2]; c = p[3]; I0 = p[4];
    foreach(levels, kap,
      forstep(T = 2, 3000, 7,
        if(A*I0^2 > K*c*T, next);
        my(s = exact(T, kap)[1], sc = smooth(T, kap), r = rho(T, kap), wc = I0 + c*sc, wd = I0 + c*s, w0 = sqrt(K*c*T/A));
        count++;
        if(s == 0, next);
        if(s == exact(T, kap)[2]*kap, boundary++);
        if(r < -10^-60 || r > 1 + 10^-60, outside++);
        identity = max(identity, abs(smooth(T - r, kap) - s));
        gap = max(gap, abs(sc - s - K*r/(A*(wc + wd) + K/kap)));
        if(sc - s < -10^-60 || sc - s >= kap, below++);
        my(exceeds = classical(T) - s > ceiling(kap), condition = c*kap*r > I0 + (w0 - wd)/2);
        if(exceeds, over++);
        if(exceeds != condition, iff++))));
  print("   ", count, " instances, ", boundary, " optima on a round boundary, ", over, " over the ceiling");
  report("rho_T in [0, 1]", outside == 0, Str(outside, " outside"));
  zero("max |S_c(T - rho_T) - S_d|", identity);
  zero("max |S_c - S_d - K rho_T/(A (w_c + w_d) + K/cap)|", gap);
  report("0 <= S_c - S_d < cap", below == 0, Str(below, " outside"));
  report("S_0 - S_d over the ceiling iff c cap rho_T > I0 + (w_0 - w_d)/2", iff == 0 && over > 0, Str(iff, " mismatches in ", over, " exceedances"));
  plant();
}

print("\n4. Table 4(a): cap = 0.03");
{
  my(kap = 3/100, rows = [
    [10^4, "18.6319", "9.7e-4", 10^-5, "0.970", "0.942654", "0.944100", "-3.8546", "-3.7573"],
    [10^5, "62.0287", "1.2e-4", 10^-5, "0.382", "0.981850", "0.982030", "-3.8579", "-3.8196"],
    [10^6, "199.3364", "4.6e-5", 10^-6, "0.455", "0.994260", "0.994328", "-3.8582", "-3.8125"],
    [10^7, "633.5653", "5.0e-6", 10^-7, "0.157", "0.998185", "0.998192", "-3.8583", "-3.8425"],
    [10^8, "2006.7252", "1.6e-6", 10^-7, "0.162", "0.999426", "0.999428", "-3.8583", "-3.8420"]]);
  zero("the ceiling is 490/729", ceiling(kap) - 490/729);
  agree("the ceiling", ceiling(kap), "0.672154", 10^-6);
  agree("B", coefficient(kap), "3.85826", 10^-5);
  agree("the band's low end, -B", -coefficient(kap), "-3.8583", 10^-4);
  agree("the band's high end, -(B - sqrt(K/(A c))/2)", -(coefficient(kap) - half_band()), "-3.7578", 10^-4);
  foreach(rows, row,
    my(T = row[1], sc = smooth(T, kap), sd = exact(T, kap)[1], r = rho(T, kap), s0 = classical(T), cl = ceiling(kap), tag = Str("T = ", T, ": "));
    agree(Str(tag, "smooth mass"), sc, row[2], 10^-4);
    agree(Str(tag, "smooth - exact"), sc - sd, row[3], row[4]);
    agree(Str(tag, "rho_T"), r, row[5], 10^-3);
    agree(Str(tag, "smooth over the ceiling"), (s0 - sc)/cl, row[6], 10^-6);
    agree(Str(tag, "exact over the ceiling"), (s0 - sd)/cl, row[7], 10^-6);
    agree(Str(tag, "smooth residual x sqrt(T)"), (s0 - sc - cl)*sqrt(T), row[8], 10^-4);
    agree(Str(tag, "exact residual x sqrt(T)"), (s0 - sd - cl)*sqrt(T), row[9], 10^-4));
  my(T = 10^8, r = rho(T, kap), off = (classical(T) - exact(T, kap)[1] - ceiling(kap))*sqrt(T) + coefficient(kap) - r*half_band());
  report("T = 1e8: the exact residual within 4e-7 of -(B - rho_T sqrt(K/(A c))/2)", abs(off) < 4*10^-7, off);
}

print("\n5. Table 4(b): T = 4000, the classical form's over-statement of the exact mass");
{
  my(T = 4000, rows = [[10^(-1/2), "0.51", 10^-2], [1/10, "1.60", 10^-2], [10^(-3/2), "5.15", 10^-2],
                       [1/100, "17.05", 10^-2], [10^(-5/2), "61.03", 10^-2], [1/1000, "236.6", 10^-1]]);
  foreach(rows, row,
    my(kap = row[1], s = exact(T, kap), sc = smooth(T, kap), s0 = classical(T));
    agree(Str("cap = ", kap, if(s[1] == s[2]*kap, ", on a boundary", ""), ": percent of the exact mass"), 100*(s0 - s[1])/s[1], row[2], row[3]);
    report(Str("cap = ", kap, ": the smooth mass's figure within 0.03 points"), abs(100*(s0 - sc)/sc - 100*(s0 - s[1])/s[1]) < 3/100, 100*(s0 - sc)/sc));
}

print("\n6. C_cap and Table 4(d): the optimum's residual after the three terms");
{
  my(rows = [
    [1/10, "-2.640763", "0.020286", "2.006e-4", "2.005734"],
    [3/100, "-6.032919", "0.049435", "4.678e-4", "4.675019"],
    [1/100, "-15.723437", "-0.033778", "-5.897e-4", "-5.921131"]]);
  foreach(rows, row,
    my(kap = row[1], tag = Str("cap = ", kap, ": "));
    agree(Str(tag, "C_cap"), constant(kap), row[2], 10^-6);
    agree(Str(tag, "residual at T = 1e4"), residual(10^4, kap), row[3], 10^-6);
    agree(Str(tag, "residual at T = 1e8"), residual(10^8, kap), row[4], 10^-7);
    agree(Str(tag, "the next coefficient"), slope(kap), row[5], 10^-6);
    print("   residual x sqrt(T), T = 1e4 .. 1e9: ", vector(6, j, residual(10^(j + 3), kap)*10^((j + 3)/2))));
}

print("\nchecks: ", checks, ", failures: ", fails);
quit;
