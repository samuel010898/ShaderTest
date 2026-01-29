#version 450

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Push {
    float width;
    float height;
    float time;
    //uint  test;
} pc;

int testIndex = 8;
vec2 resolution = vec2(pc.width, pc.height);

vec2 pixelCoord()
{
    return gl_FragCoord.xy;
}

vec2 uv()
{
    return gl_FragCoord.xy / resolution; //0-1
}
// center at 0,0shortest axis -1,1
vec2 screenCoord()
{
    vec2 p = (gl_FragCoord.xy / resolution) * 2.0 - 1.0;
    float aspect = resolution.x / resolution.y;
    p.x *= aspect;
    return p;
}

vec3 test_axes()
{
    vec2 p = screenCoord();
    float lineX = smoothstep(0.01, 0.0, abs(p.y));
    float lineY = smoothstep(0.01, 0.0, abs(p.x));
    return vec3(lineX + lineY);
}

vec3 test_diagonals()
{
    vec2 p = screenCoord();
    float d1 = smoothstep(0.01, 0.0, abs(p.x - p.y));
    float d2 = smoothstep(0.01, 0.0, abs(p.x + p.y));
    return vec3(d1, d2, 0.0);
}

vec3 test_circle()
{
    vec2 p = screenCoord();
    float r = length(p);
    float circle = smoothstep(0.02, 0.0, abs(r - 0.75));
    return vec3(circle);
}

vec3 test_bounds()
{
    vec2 p = screenCoord();
    float inside =
        step(-1.0, p.x) * step(p.x, 1.0) *
        step(-1.0, p.y) * step(p.y, 1.0);

    return mix(vec3(1,0,0), vec3(0,1,0), inside);
}

vec3 test_grid()
{
    vec2 p = screenCoord() * 10.0;
    vec2 g = abs(fract(p) - 0.5);
    float line = step(min(g.x, g.y), 0.02);
    return vec3(line);
}

vec3 test_pixel_centers()
{
    vec2 f = fract(gl_FragCoord.xy);
    float dot = step(length(f - 0.5), 0.1);
    return vec3(dot);
}

vec3 test_uv()
{
    vec2 u = uv();
    return vec3(u.x, u.y, 0.0);
}

vec3 test_distance()
{
    float d = length(screenCoord());
    return vec3(d);
}

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

vec3 randColor(vec2 p) {
    return vec3(
        hash(p + 0.1),
        hash(p + 17.3),
        hash(p + 42.9)
    );
}

double remap(double t, double min1, double max1, double min2, double max2)
{
	if (min1 >= max1) {return min2;}
	return min2 + (max2 - min2) * ((t - min1) / (max1 - min1));
}

vec3 brightnessToColor(float v, float min, float max)
{
	if (max <= min) return vec3(1,0,0);
	if (v < min) return vec3(1.0,1.0,0.0); //yellow
	if (v > max) return vec3(0.0,1.0,1.0); //cyan
	float x = (v - min) / (max - min);
	float t = x * 7.0;
    	int seg = int(t);
    	float f = t - seg;
	//f = f * f * (3.0f - 2.0f * f); // smoothing
	float r=0, g=0, b=0;
    	switch (seg)
	{
        	case 0: b=(f); break; //black→blue
        	case 1: b=1.0; g=((f)); break; //blue→cyan
        	case 2: g=1.0; b=1.0-f; break; //cyan→green
        	case 3: g=1.0; r=((f)); break; //green→yellow
        	case 4: r=1.0; g=1.0-f; break; //yellow→red
        	case 5: r=1.0; b=((f)); break; //red→magenta
		case 6: r=1.0; g=((f)); b=1.0; break; //magenta→white
		default: return randColor(vec2(v,v-((720/2)/480/2)+720/2)); //overflow
    	}
    	return vec3(r, g, b);
}
int calla(int a);
int callb(int b);
int callc(int c);
int calld(int d);
int calle(int e);
int callf(int f);
int calla(int a){
	return callb(a);
}
int callb(int b){
	return callc(b);
}
int callc(int c){
	return calld(c);
}
int calld(int d){
	return calle(d);
}
int calle(int e){
	return callf(e);
}
int callf(int f){
	return calla(f);
}


void main()
{
    	vec3 color;

	/*switch(testIndex){
    		case 0:color = test_axes(); break;
    		case 1:color = test_diagonals(); break;
    		case 2:color = test_circle(); break;
    		case 3:color = test_bounds(); break;
    		case 4:color = test_grid(); break;
    		case 5:color = test_pixel_centers(); break;
    		case 6:color = test_uv(); break;
    		case 7:color = test_distance(); break;
		default: color = vec3(1,0,1); break;
	}*/
	
	
	outColor = vec4(1.0,1.0,1.0, 1.0);
	//outColor = vec4(brightnessToColor((screen.x)*(screen.y)+screen.x, -1.0, 1.0), 1.0);

    	//outColor = vec4(color, 1.0);
}


		/*case 8:{
			vec2 s = vec2(pc.time, hash(vec2(pc.time, pc.time)));
			vec2 n, l = vec2(hash(vec2(pc.time, pc.time)), pc.time);
			for(double i=0;i<(pc.height*pc.width+pc.width);i++){
				
				n = vec2(hash(s), hash(s));
				s = vec2(hash(l), hash(l));
				l = vec2(hash(n), hash(n));
			}
			s += n + l;
			color = randColor(screen * 100.0 + s);
			break;
		}*/
