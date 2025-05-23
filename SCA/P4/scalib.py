"""

Biblioteca con funciones y clases usadas para las prácticas de la asginatura de
Sistemas de Codificación y Almacenamiento.

Fermín Segovia Román
Dpt. de Teoría de la Señal, Telemática y Comunicaciones
Universidad de Granada

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn.cluster


# %% Cuantificadores


class UniformSQ:
    """
    Construye un cuantificador escalar uniforme

    Atributos:
        q: float
            Tamaño del cuanto

        C: numpy.array
            Array de numpy con todos los niveles de cuantificación

    """

    q, L, C = None, None, None
    precision = None
    roundData = lambda x: None

    def __init__(self, b, inputRange, qtype="midrise", precision=4):
        """
        Constructor de la clase

        Args:
            b: int
                Tasa del bits por muestra a la salida del cuantificador

            inputRange: tuple
                Rango de entrada del cuantificador. Debe ser una tupla con dos
                valores, los límites inferior y superior del rango.

            qtype: {'midrise', 'midtread'} (por defecto: 'midrise')
                Tipo del cuantificador (media contrahuella o media huella)

        """

        self.precision = precision
        self.xMin, self.xMax = inputRange
        if qtype == "midrise":
            self.L = 2**b
            self.q = (float(self.xMax) - float(self.xMin)) / self.L
            self.roundData = lambda x: np.floor((x - self.xMin) / self.q)
        elif qtype == "midtread":
            self.L = 2**b - 1
            self.q = (float(self.xMax) - float(self.xMin)) / self.L
            self.roundData = lambda x: np.round((x - self.xMin) / self.q - 0.5)
        else:
            print("Invalid type:", qtype)

        self.C = np.arange(self.xMin + self.q / 2, self.xMax, self.q)
        if self.precision is not None:
            self.C = np.round(self.C, self.precision)

    def quantize(self, data):
        """
        Cuantifica una señal

        Args:
            data: numpy.array
                Array de numpy con las muestras de la señal

        Return:
            Array de numpy con las muestras de la señal cuantificadas. Tendrá
            las mismas dimensiones que 'data'

        """

        code = self.encode(data)
        return self.decode(code).astype(data.dtype).reshape(data.shape)

    def encode(self, data):
        """
        Realiza el mapeo del codificador (primera parte de la cuantificación)
        correspondiente a 'data'

        Args:
            data: numpy.array
                Array de numpy con las muestras de la señal

        Return:
            Array de numpy con el código correspondiente a 'data'

        """

        code = self.roundData(np.array(data, dtype=float)).astype(int).flatten()
        code = np.clip(code, 0, self.L - 1)
        return code

    def decode(self, code):
        """
        Realiza el mapeo del descodificador (segunda parte de la
        cuantificación).

        Args:
            code: numpy.array
                Array de numpy con las muestras de la señal codificadas

        Return:
            Array de numpy con las muestras cuantificadas de la señal
            codificada en 'code'.

        """

        dataQ = (code + 0.5) * self.q + self.xMin
        if self.precision is not None:
            dataQ = np.round(dataQ, self.precision)
        return dataQ

    def plot(self, ax=None):
        """
        Representa la entrada frente a la salida del cuantificador

        Args:
            ax: Axes, opcional (por defecto: None)
                Ejes cartesianos en los que se representará el cuantificador.
                Si es None, se crearán unos nuevos ejes en una nueva figura.

        """

        data = np.linspace(self.xMin, self.xMax, 1000)
        dataQ = self.quantize(data)

        if ax is None:
            fig, ax = plt.subplots(layout="tight")
        ax.plot(data, dataQ)
        ax.set(xlabel="Entrada", ylabel="Salida")
        ax.grid("on")


class OptimalVQ:
    """
    Construye un cuantificador vectorial optimizado a la PDF de la señal y
    basado en el algoritmo de Lloyd-Max

    Atributos:
        C: numpy.array
            Array de numpy con todos los niveles de cuantificación

    """

    C = None

    def __init__(self, b, data, algorithm="kmeans"):
        """
        Constructor de la clase

        Args:
            b: int
                Tasa del bits por muestra a la salida del cuantificador

            data: numpy.array
                Datos de entrenamiento usados para definir los intervalos de
                cuantificación óptimos mediante el algoritmo de Lloyd-Max.
                Es un array de numpy de 2 dimensiones con las muestras de la
                señal. La primera dimensión indica el número de bloques
                (vectores que serán tratados como un solo objeto por el
                cuantificador vectorial) y la segunda dimensión es el tamaño
                del bloque (N).

            algorithm: 'kmeans' o función, opcional (por defecto: 'kmeans')
                Algorimo para crear el conjunto de niveles de cuantificación.
                Por defecto se usa la clase KMeans de sklearn pero si este
                argumento es una función se llamará a esa función. La función
                debe aceptar dos argumenos de entrada: 'b' y 'data' y debe
                devolver un array de Numpy con el conjunto de de niveles de
                cuantificación.

        """

        if algorithm == "kmeans":
            obj = sklearn.cluster.KMeans(2**b, n_init="auto").fit(data)
            C = obj.cluster_centers_
        elif callable(algorithm):
            C = algorithm(b, data)
        else:
            print("Wrong algorithm")

        self.C = np.squeeze(np.sort(C, axis=0))

    def quantize(self, data):
        """
        Cuantifica una señal

        Args:
            data: numpy.array
                Array de numpy de 2 dimensiones con las muestras de la señal.
                La primera dimensión indica el número de bloques (vectores que
                serán cuantificados como un solo objeto por el cuantificador
                vectorial) y la segunda dimensión es el tamaño del bloque (N).

        Return:
            Array de numpy con las muestras de la señal cuantificadas. Tendrá
            las mismas dimensiones que 'data'

        """

        code = self.encode(data)
        return self.decode(code).astype(data.dtype)

    def encode(self, data):
        """
        Realiza el mapeo del codificador (primera parte de la cuantificación)
        correspondiente a 'data'

        Args:
            data: numpy.array
                Array de numpy con las muestras de la señal

        Return:
            Array de numpy con el código correspondiente a 'data'

        """

        idx = np.zeros(len(data), dtype=int)
        for j in range(len(data)):
            distance = np.abs(self.C - data[j])
            if distance.ndim > 1:
                distance = np.sum(distance, axis=1)
            idx[j] = np.argmin(distance)
        return idx

    def decode(self, code):
        """
        Realiza el mapeo del descodificador (segunda parte de la
        cuantificación).

        Args:
            code: numpy.array
                Array de numpy con las muestras de la señal codificadas

        Return:
            Array de numpy con las muestras cuantificadas de la señal
            codificada en 'code'.

        """
        return self.C[code.astype(int)]


class OptimalSQ(OptimalVQ):
    """
    Construye un cuantificador escalar optimizado a la PDF de la señal y
    basado en el algoritmo de Lloyd-Max

    Atributos:
        C: numpy.array
            Array de numpy con todos los niveles de cuantificación

    """

    def __init__(self, b, data, algorithm="kmeans"):
        """
        Constructor de la clase

        Args:
            b: int
                Tasa del bits por muestra a la salida del cuantificador

            data: numpy.array
                Datos de entrenamiento usados para definir los intervalos de
                cuantificación óptimos mediante el algoritmo de Lloyd-Max.

            algorithm: 'kmeans' o función, opcional (por defecto: 'kmeans')
                Algorimo para crear el conjunto de niveles de cuantificación.
                Por defecto se usa la clase KMeans de sklearn pero si este
                argumento es una función se llamará a esa función. La función
                debe aceptar dos argumenos de entrada: 'b' y 'data' y debe
                devolver un array de Numpy con el conjunto de de niveles de
                cuantificación.

        """

        super().__init__(b, data.reshape(-1, 1), algorithm)

    def quantize(self, data):
        """
        Cuantifica una señal

        Args:
            data: numpy.array
                Array de numpy con las muestras de la señal

        Return:
            Array de numpy con las muestras de la señal cuantificadas. Tendrá
            las mismas dimensiones que 'data'

        """

        dataQ = super().quantize(data.reshape(-1, 1))
        return dataQ.reshape(data.shape)

    def plot(self, ax=None):
        """
        Representa la entrada frente a la salida del cuantificador

        Args:
            ax: Axes, opcional (por defecto: None)
                Ejes cartesianos en los que se representará el cuantificador.
                Si es None, se crearán unos nuevos ejes en una nueva figura.

        """

        q = np.diff(self.C).mean()
        data = np.linspace(self.C[0] - q, self.C[-1] + q, 1000)
        dataQ = self.quantize(data)

        if ax is None:
            fig, ax = plt.subplots(layout="tight")
        ax.plot(data, dataQ)
        ax.set(xlabel="Entrada", ylabel="Salida")
        ax.grid("on")


# %% Codificador de la fuente


class FixedLengthCoder:
    """
    Construye un codificador de palabras de longitud fija

    Atributos:
        b: int
            Número de bits de cada palabra. Debe ser un entero positivo (>=0)

    """

    b = 1

    def __init__(self, b):
        """
        Constructor de la clase

        Args:
            b: int
                Número de bits de cada palabra. Debe ser un entero positivo (>=0)

        """

        self.b = min(max(int(b), 1), 64)

    def encode(self, data):
        """
        Codifica un mensaje

        Args:
            data: numpy.array
                Array de numpy con el mensaje a codificar. Sus elementos deben
                ser números enteros mayores o iguales a 0 y menores a 2**b

        Return:
            Array de numpy con la cadena de bits correspondiente a la
            codificacióin de 'data'

        """

        if np.array(data).ndim == 0:
            data = np.array([data])
        code = [*"".join([f"{x:0{self.b}b}" for x in data])]
        return np.array(code, dtype="uint8")

    def decode(self, code):
        """
        Descodifica un mensaje

        Args:
            code: numpy.array
                Array de numpy con 0s y 1s (secuencia de bits).

        Return:
            Array de numpy con el mensaje codificado en 'code'

        """

        return np.array(
            [
                int("".join(code[i : i + self.b].astype("<U1")), 2)
                for i in range(0, len(code), self.b)
            ]
        )


# %% Funciones de utilidad


def signalRange(s):
    """
    Detemina el rango en el que varia una señal en función del tipo de dato del
    array en el que está almacenada

    Args:
        s: numpy.array
            Señal de la que se determinará el rango

    Return:
        Tupla de 2 elementos con el límite inferior y el límite superior del
        rango de la señal 's'.
    """

    if np.issubdtype(s.dtype, np.integer):
        info = np.iinfo(s.dtype)
        return (info.min, info.max + 1)
    elif np.min(s) >= 0:
        return (0, 1)
    else:
        return (-1, 1)


def toDB(power, pref=1):
    """
    Convierte un valor de potencia en decibelios

    Args:
        power: float
            Valor de potencia a convertir en decibelios

        pref: float, opcional (por defecto: 1)
            Valor de referencia

    Return:
        Valor 'power' en decibelios
    """

    return np.round(10 * np.log10(power / pref), 4)


def genDither(nSamples, q, pdf, dtype=float):
    """
    Genera dither con una PDF y potencia determinada

    Args:
        nSamples: int
            Número de muestras que se generarán

        q:
            Valor del cuanto del cuantificador para el que se va a usar el
            dither generado. Este valor determina la potencia del dither

        pdf: {'gaussian', 'rectangular', 'triangular'}
            PDF del dither que se va a generar

        dtype: dtype, opcional (por defecto: float)
            Tipo de datos de las muestras del dither

    Return:
        Array de numpy con las muestras del dither generado
    """

    if pdf == "triangular":
        dither = np.random.triangular(-q, 0, q, nSamples)
    elif pdf == "rectangular":
        dither = np.random.uniform(-q / 2, q / 2, nSamples)
    elif pdf == "gaussian":
        dither = np.random.normal(0, q / 2, nSamples)
    else:
        print("Invalid PDF type:", pdf)
        dither = np.zeros(nSamples)
    return dither.astype(dtype)


def addDither(x, dither):
    """
    Añade dither a una señal evitando el desbordamiento del tipo de datos

    Args:
        x: numpy.array
            Señal a la que se va a añadir dither

        dither: numpy.array
            Dither que se va a añadir a 'x'

    Return:
        Señal 'x' con el dither añadido

    """

    xMin, xMax = signalRange(x)
    xDither = np.clip(x.astype(float) + dither, xMin, xMax)
    return xDither.astype(x.dtype)


def getNumberOfBits(s):
    """
    Aproxima el número de bits que se usaron para cuantificar una señal

    Args:
        s: numpy.array
            Senal de la que se van a determinar los bits usados en su
            cuantificación

    Return:
        Número de bits aproximado (cota inferior) que se usó para
        cuantificar 's'
    """

    return int(np.ceil(np.log2(len(np.unique(s)))))


def mse(s1, s2):
    """
    Calcula el error cuadrático medio (MSE) entre 2 señales

    Args:
        s1: numpy.array
            Primera señal

        s2: numpy.array
            Segunda señal

    Return:
        MSE entre 's1' y 's2'

    """

    return np.mean((s1 - s2) ** 2)


def snr(s1, s2):
    """
    Calcula la relación señal ruido (SNR) en decibelios entre 2 señales

    Args:
        s1: numpy.array
            Primera señal

        s2: numpy.array
            Segunda señal

    Return:
        SNR entre 's1' y 's2'

    """
    s1 = s1.flatten().astype(float)
    s2 = s2.flatten().astype(float)
    n = np.min((len(s1), len(s2)))
    pSig = np.sum(s1[:n] ** 2)
    pErr = np.sum((s1[:n] - s2[:n]) ** 2)
    return 10 * np.log10(pSig / pErr)


def psnr(s1, s2):
    """
    Calcula la relación señal ruido pico (PSNR) en decibelios entre 2 señales

    Args:
        s1: numpy.array
            Primera señal

        s2: numpy.array
            Segunda señal

    Return:
        PSNR entre 's1' y 's2'

    """
    pSig = np.abs(s1).max() ** 2.0
    pErr = np.mean((s1.flatten().astype(float) - s2.flatten().astype(float)) ** 2)
    return 10 * np.log10(pSig / pErr)


def partitionImage(img, step1, step2):
    """
    Divide una imagen bidimensional en bloques rectangulares. Completa la
    imagen con ceros si es necesario.

    Args:
        img: numpy.array
            Imagen que se va a dividir en bloques

        step1: int
            Alto (en pixeles) de los bloques en los que se va a dividir 'img'

        step2: int
            Ancho (en pixeles) de los bloques en los que se va a dividir 'img'

    Return:
        Array de numpy con los pixels de cada bloque en forma de vector 1D
        Tiene dimensión AxB, donde A es el número de bloques y B es el número
        de píxeles en un bloque (B='step1'*'step2')
    """

    # Asegura imagen de 2 dimensiones
    if img.ndim == 3:
        img = img.reshape(img.shape[0], -1, order="F")

    # Ajusta filas y cols para que sean divisibles por step1 y step2
    nRow, nCol = img.shape
    padRow = (step1 - nRow % step1) % step1
    padCol = (step2 - nCol % step2) % step2
    img = np.pad(img, [(0, padRow), (0, padCol)], mode="edge")

    # Particiona en bloques de step1 x step2
    blocks = []
    for r in range(0, nRow, step1):
        for c in range(0, nCol, step2):
            blocks.append(img[r : r + step1, c : c + step2].flatten())

    return np.array(blocks)


def composeImage(blocks, step1, step2, finalShape):
    """
    Reconstruye una imagen bidimensional que previamente se ha divivido en
    bloque usando 'partition image'.

    Args:
        blocks: numpy.array
            Array de tamaño AxB con los bloques de la imagen, donde A es el
            número de bloques y B es el número de píxeles en un bloque

        step1: int
            Alto (en pixeles) de los bloques en los que se dividió 'img'

        step2: int
            Ancho (en pixeles) de los bloques en los que se dividió 'img'

        finalShape: tuple
            Tamaño de la imagen tras la reconstrucción.

    Return:
        Array de numpy con la imagen reconstruida.

    """

    nRow, nCol = finalShape[:2]
    if len(finalShape) == 3:
        nCol *= finalShape[2]

    bRow, bCol = np.ceil((nRow / step1, nCol / step2))
    img = np.zeros((int(bRow * step1), int(bCol * step2)), dtype=blocks.dtype)

    r, c = 0, 0
    for i in range(len(blocks)):
        block = blocks[i].reshape((step1, step2))
        img[r : r + step1, c : c + step2] = block
        c += step2
        if c >= nCol:
            r, c = r + step1, 0

    return img[:nRow, :nCol].reshape(finalShape, order="F")


def dct(N):
    """
    Construye una matriz de transformación DCT de tamaño N x N

    Args:
        N: int
            Número de filas/columnas de la DCT

    Return:
        Matriz de transformación DCT

    """

    return scipy.fft.dct(np.eye(N), axis=0, norm="ortho")


def dDCT(data, C=None):
    """
    Aplica la DCT a un bloque de datos bidimensional

    Args:
        data: numpy.array
            Bloque de datos al que aplica la DCT

        C: numpy.array, opcional
            Matriz de transformación DCT

    Return:
        Array de numpy con los coeficientes DCT correspondientes a 'data'

    """

    return scipy.fft.dctn(data, norm="ortho")


def iDCT(coef, C=None):
    """
    Aplica la DCT inversa a un bloque de coeficientes DCT bidimensional

    Args:
        coef: numpy.array
            Bloque de coeficientes DCT al que aplica la DCT inversa

        C: numpy.array, opcional
            Matriz de transformación DCT

    Return:
        Array de numpy con los datos correspondientes a 'coef'

    """

    return scipy.fft.idctn(coef, norm="ortho")


def zigzag(m):
    """
    Recorre un array 2D en forma de zig-zag

    Args:
        m: numpy.array
            Array 2D que va a se recorrido en zig-zag

    Return:
        Array 1D de numpy con los elementos de m leídos en zig-zag

    """

    h, w = m.shape
    zz = []
    for s in range(w + h + 1):
        for i in range(h)[:: s % 2 * 2 - 1]:
            if -1 < s - i < w:
                zz.append(m[i, s - i])
    return np.array(zz)


def zigzagIdx(h, w):
    """
    Recorre un array 2D en forma de zig-zag

    Args:
        h: int
            Número de filas del array 2D

        w: int
            Número de columnas del array 2D

    Return:
        Tupla con los índices correspondientes a recoger un array 2D
        con h filas y w columnas en zig-zag

    """

    idxr, idxc = [], []
    for s in range(w + h + 1):
        for i in range(h)[:: s % 2 * 2 - 1]:
            if -1 < s - i < w:
                idxr.append(i)
                idxc.append(s - i)
    return (idxr, idxc)


def downsample(s, n):
    """
    Decrementa la frecienca de muestreo un factor n

    Args:
        s: numpy.array
            Señal unidimensional cuya frecuencia de muestreo va a ser
            decrementada.

        n: int
            Factor de decremento de la frecuencia de muestreo.

    Return:
        Señal 's' con la frecienca de muestreo decrementada un factor n

    """
    return s[::n]


def upsample(s, n):
    """
    Aumenta la frecienca de muestreo un factor n

    Args:
        s: numpy.array
            Señal unidimensional cuya frecuencia de muestreo va a ser
            aumentada.

        n: int
            Factor de aumento de la frecuencia de muestreo.

    Return:
        Señal 's' con la frecienca de muestreo aumentada un factor n

    """
    ss = np.zeros(len(s) * n, dtype=s.dtype)
    ss[::n] = s
    return ss


def bode(b, a=1):
    """
    Calcula y representa la respuesta en frecuencia de un sistema

    Args:
        b: numpy.array
            Coeficientes del numerador de la función de transferencia
        a: numpy.array (opcional, por defecto: 1)
            Coeficientes del denominador (en sistemas FIR es 1)

    Return:
        None
    """

    w, H = scipy.signal.freqz(b, a)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), layout="tight")
    mag = 20 * np.log10(np.abs(H))
    mRange = [np.min((-10, np.min(mag))), np.max((3, np.max(mag)))]

    ax1.plot(w / np.pi, mag)
    ax1.set(ylabel="Magnitud (dB)", xlim=[0, 1])
    if not np.any(np.isinf(mRange)) and not np.any(np.isnan(mRange)):
        ax1.set(ylim=mRange)
    ax1.grid()

    ax2.plot(w / np.pi, np.unwrap(np.angle(H)) * 180 / np.pi)
    ax2.set(xlabel="Frecuencia normalizada ($x \\pi$ rad/muestra)")
    ax2.set(ylabel="Fase (grados)", xlim=[0, 1])
    ax2.grid()


def filterDelay(b, a=1):
    """
    Calcula y representa el retardo de grupo de un sistema, y devuelve su valor medio

    Args:
        b: numpy.array
            Coeficientes del numerador de la función de transferencia
        a: numpy.array, opcional (por defecto: 1)
            Coeficientes del denominador

    Return:
        float
            Retardo medio en muestras
    """

    w, d = scipy.signal.group_delay((b, a))

    # Mostrar gráfica
    fig, ax = plt.subplots(layout="tight", figsize=(12, 5))
    ax.plot(w / np.pi, np.round(d))
    ax.set(xlabel="Frecuencia normalizada ($x \pi$ rad/muestra)")
    ax.set(ylabel="Retardo (muestras)", xlim=[0, 1])

    # Devolver valor medio del retardo
    return np.mean(d)


def getG722lpf():
    """
    Devuelve los coeficientes del filtro paso-baja de análisis definido en el
    estándar G.722

    Args:
        (No tiene)
    Return:
        Array de numpy con los coeficientes de la respuesta al impulso del
        filtro

    """

    h1 = [
        0.366211e-3,
        -0.134277e-2,
        -0.134277e-2,
        0.646973e-2,
        0.146484e-2,
        -0.190430e-1,
        0.390625e-2,
        0.44189e-1,
        -0.256348e-1,
        -0.98266e-1,
        0.116089,
        0.473145,
    ]
    h1 = h1 + h1[::-1]
    return np.array(h1)


# %% Codificador del canal


class CRC:
    """
    Calcula y comprueba códigos de redundancia cíclica.

    Atributos:
        p: numpy.array
            Polinomio generador

    """

    p = None

    def __init__(self, polgen):
        """
        Contructor de la clase.

        Args:
            poligen: iterable
                Polinomio generador. El primer elemento corresponde al
                coeficiente de mayor grado.
        """

        self.p = np.array(polgen)

    def register(self, bits):
        """
        Función interna. Opera el registro de desplazamiento

        """

        r = np.zeros(len(self.p) - 1, dtype="uint8")

        for i in range(len(bits)):
            r = np.roll(r, -1)
            last = r[-1]
            r[-1] = bits[i]
            if last:
                r = np.logical_xor(r, self.p[1:])

        return r

    def encode(self, bits):
        """
        Genera una palabra de código. Para ello calcula el CRC corresponediente
        a una palabra de mensaje y se lo añade al final.

        Args:
            bits: numpy.array
                Palabra de mensaje

        Return:
            Array de numpy con la palabra de código
        """

        bitsAll = np.concatenate((bits, np.zeros(len(self.p) - 1, dtype="uint8")))
        crc = self.register(bitsAll)
        return np.concatenate((bits, crc))

    def decode(self, bits):
        """
        Comprueba una palabra de código y determina si ha habido o no errores.

        Args:
            bits: numpy.array
                Palabra de mensaje

        Return:
            Tupla con un array de numpy con la palabra de mensaje y un valor
            booleano indicando si ha habido o no errores

        """

        msg = bits[: -(len(self.p) - 1)]
        syndrome = self.register(bits)
        err = bool(np.sum(syndrome))
        return msg, err


# %%


def canalBSC(bits, ber):
    """
    Simula la transmisión de datos por un canal BSC.

    Args:
        bits: numpy.array
            Array unidimensional con valores '0' o '1' que contiene los datos
            transmitidos por el canal.

        ber: float
            Probabilidad de error de bits del canal. Por ejemplo, un valor de
            0.05 equivale a un BER del 5%

    Return:
        Array de numpy con los datos de 'bits' tras la transmisión por el canal

    """

    idx = np.random.random(len(bits)) <= ber
    output = np.copy(bits)
    output[idx] = (output[idx] + 1) % 2
    return output
