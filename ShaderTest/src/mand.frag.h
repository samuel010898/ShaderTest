// mand.frag.h
vec3 viridis(float t);
vec3 plasma(float t);
vec3 magma(float t);
vec3 hot(float t);
vec3 coolwarm(float t);
vec3 twilight(float t);
vec3 binary(float t);
vec3 histogramEqualized(float t);
float tricorn();
float mandOrbitTrap();
float multibrot(float power);
/*float perpMand()
{
    const double ESCAPE2 = 16.0;

    dvec2 c = windowCoordD();
    dvec2 z = dvec2(0.0);

    int i;
    for (i = 0; i < MAX_ITER; i++) {
        z = dvec2(z.x * z.x - z.y * z.y + c.x, -2.0 * z.x * z.y + c.y);
        if (dot(z, z) > ESCAPE2) break;
    }

    return float(i) / float(MAX_ITER);
}*/
float celticMand();
float orbitAngle();
float boxTrap();
float ikeda();
float henon();
float lissajous();
float lyapunov();
float collatz();
float biomorph();
float phoenix();
float magnet1();
float nova();
float tetration();
float gingerbreadman();
float duffing();
float tinkerbell();
float gumowski_mira();
float peter_de_jong();
float clifford();
float hopalong();
float apollonian_gasket();
float kleinian();
float fractal_noise();
float worley();