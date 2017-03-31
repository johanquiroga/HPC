# Informe

## I.  Introducción

En el presente informe se busca presentar la implementación del algoritmo de multiplicación de matrices utilizando CUDA y aprovechando la memoria compartida disponible para cada bloque de hilos. También se muestran algunos resultados obtenidos de pruebas de rendimiento que se realizaron.

## II.  Desarrollo del tema

Hasta ahora se venía trabajando con un kernel de CUDA (ver [Figura 1][fig1]) el cual era ejecutado por una gran cantidad de hilos al mismo tiempo, cada uno calculando un único valor de la matriz resultado. Para esto se debían copiar previamente los datos de las matrices que se iban a multiplicar a la memmoria global del dispositivo, una vez copiados se procedía con el procesamiento de estos datos (lanzamiento del kernel).
```c
__global__ void matrixMultDevice(float* d_A, float* d_B, float* d_C, int width) {
	int Row = blockDim.y * blockIdx.y + threadIdx.y;
	int Col = blockDim.x * blockIdx.x + threadIdx.x;
	if(Row < width && Col < width) {
		float ans = 0.0;
		for(int k=0; k<width; k++) {
			ans += d_A[Row*width+k]*d_B[k*width+Col];
		}
		d_C[Row*width+Col]=ans;
	}
}
``` 
#### *Figura 1: Kernel multiplicación de matrices*

Esta es una buena primera solución a la multiplicación de matrices, sin embargo no es la más deseada ya que presenta algunos inconvenientes. Entre los más destacados, es que al guardar los datos en memoria global, el acceso a los mismos es muy lento y por lo tanto no se logra aprovecar todo el potencial del hardware del dispositivo.

Para solucionar esto se propone el uso de los distintos niveles de memoria con los que cuentan las GPU; más especificamente se busca aprovechar la memoria compartida con la que cuenta cada bloque de hilos. Esta memoria tiene como principal característica que puede ser accedida a una velocidad muy alta y permite un accesso totalmente paralelo, es decir, todos los hilos que tengan acceso a ella lo pueden hacer en cualquier momento de su ejecución sin ningún bloqueo o inconveniente parecido.
Con la memoria compartida hay que tener en cuenta que solamente está disponible para los hilos en un mismo bloque, y cada bloque tiene su propia memoria.
La principal desventaja de esta memoria es que normalmente cuenta con una capacidad de almacenamiento muy baja, por lo que, cuando se vaya a utilizar hay que tener en cuenta que los datos que se vayan a cargar quepan en dicha memmoria.

La estrategia que se toma para sobrellevar esta desventaja es particionar los datos en subconjuntos, llamados *tiles*. Estos *tiles* podrán ser cargados en memoria compartida y procesados en el kernel independientes unos de otros. Este proceso de cargar y procesar los datos lo van a realizar todos los hilos del bloque paralelamente, tal como funcionaba en la versión del kernel anterior.


## III.  Resultados

### **Gráfica de Tiempos:**
![Grafica de tiempos](imgs/grafica_tiempos.png)  
*Grafica 1: tamaño de matriz contra tiempo de ejecución*

### **Gráfica de Aceleración:**
![Grafica de aceleracion](imgs/grafica_aceleracion.png)  
*Grafica 2: tamaño de matriz contra aceleración obtenida*

## IV.  Conclusiones



[fig1]: ####figura-1:-kernel-multiplicacion-de-matrices