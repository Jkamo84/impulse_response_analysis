import numpy as np                                  #para utilizar señales como arrays multidimensionales
import matplotlib.pyplot as plt
import soundfile as sf                              #para realizar lectura de archivos .wav

import sounddevice as sd                            #para grabar audio (no fué implementado en este notebook)
from scipy.io.wavfile import write                  #para realizar escritura de archivos .wav
from scipy import signal                            #para realizar operaciones complejas sobre senales
from scipy.signal import butter, lfilter, freqz     #para crear filtros y aplicarlos
from scipy import integrate                         #para realizar integración sobre senales

#from playsound import playsound
from IPython.display import Audio                   #para reproducir audio generando componentes de navegador

from scipy.fftpack import fft                       #para realizar transformada de fourier sobre senales

# %matplotlib inline

data, samplerate = sf.read('/content/fluctuating.wav')  #lectura de archivo, obteniendo array y frecuencia de muestreo de la misma
l = len(data)
t1 = np.linspace(0, l / samplerate, l)                  #generar un array de tiempo basado en el largo de la senal y su samplerate
x = data / np.max(np.abs(data))                         #normalización de la señal

data, samplerate = sf.read('/content/reverb.wav')
l = len(data)
t2 = np.linspace(0, l / samplerate, l)
y = data / np.max(np.abs(data))

data, samplerate = sf.read('/content/multiple.wav')
l = len(data)
t3 = np.linspace(0, l / samplerate, l)
z = data / np.max(np.abs(data))

data, samplerate = sf.read('/content/hey.wav')
l = len(data)
t4 = np.linspace(0, l / samplerate, l)
a = data / np.max(np.abs(data))

fig = plt.figure()

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.plot(t1, x, 'r')
ax1.set_ylabel('amplitud')
ax1.set_xlabel('tiempo [s]')
ax1.set_title('Eco fluctuante')

ax2.plot(t2, y, 'b')
ax2.set_ylabel('amplitud')
ax2.set_xlabel('tiempo [s]')
ax2.set_title('Recinto reverberante')

ax3.plot(t3, z, 'g')
ax3.set_ylabel('amplitud')
ax3.set_xlabel('tiempo [s]')
ax3.set_title('Multiples impulsos')

ax4.plot(t4, a, 'k')
ax4.set_ylabel('amplitud')
ax4.set_xlabel('tiempo [s]')
ax4.set_title('Un "Hey!"')

fig.tight_layout()
plt.show()

# %matplotlib inline
fs = 44100
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

def fft_chunks(signal,fs,empty_matrix, axes):
    '''
    función para aplicar FFT a trozos de senal distribuidos homogeneamente a lo largo del tiempo
    regresa una matríz tiempo vs frecuencia vs amplitud
    '''  
    signal_matrix = empty_matrix                        #matríz vacía para ser llenada con la transformada de cada trozo de senal
    chunk_fs = 1000                                     #numero de muestras que tendrá cada trozo de senal
    s_max = 0                                           #variable que almacena mayor amplitud para luego normalización

    for i in range(0,len(signal) - chunk_fs,chunk_fs):  #recorrido por cada trozo de senal
        s = fft(signal[i:i + chunk_fs])                 #transformada sencilla del trozo de senal
        s_abs = np.abs(s[:int(len(s)/2)])               #utilizar la mitad del array resultante (espejo) y convertir números imaginarios en magnitudes
        signal_matrix[int(i / chunk_fs)] = s_abs        #almacenar resultado
        s_max = np.max([s_max,np.max(s_abs)])           #encontrar el dato de mayor amplitud

    signal_matrix /= s_max                              #normalización de la matríz resultante

    #plot de matríz como mapa de calor en los rangos de tiempo y de frecuencia [Hz]
    axes.imshow(signal_matrix.transpose(), cmap='hot', interpolation='nearest', aspect='auto', extent=[0,1,22050,0])
    #axes.set_yscale('log')
    return signal_matrix

#cada senal tiene una matríz vacía diferente porque no todas tienen la misma longitud
rt_x = fft_chunks(x[:,0], fs, np.empty([20,500]), ax1)
rt_y = fft_chunks(y[:,0], fs, np.empty([100,500]), ax2)
rt_z = fft_chunks(z[:,0], fs, np.empty([40,500]), ax3)
rt_a = fft_chunks(a, fs, np.empty([40,500]), ax4)

#se asignan limites de frecuencia de 0 a 10000 porque no hay información útil de ese número en adelante
ax1.set_ylim((10000, 0)) 
ax1.set_xlim((0, 0.6)) 
ax1.set_ylabel('frecuencia [Hz]')
ax1.set_xlabel('tiempo [s]')
ax1.set_title('Eco fluctuante')

ax2.set_ylim((10000, 0)) 
ax2.set_xlim((0, 0.6)) 
ax2.plot(t2, y, 'b')
ax2.set_ylabel('frecuencia [Hz]')
ax2.set_xlabel('tiempo [s]')
ax2.set_title('Recinto reverberante')

ax3.set_ylim((10000, 0)) 
ax3.set_xlim((0, 0.6))
ax3.set_ylabel('frecuencia [Hz]')
ax3.set_xlabel('tiempo [s]')
ax3.set_title('Multiples impulsos')

ax4.set_ylim((10000, 0)) 
ax4.set_xlim((0, 0.6)) 
ax4.set_ylabel('frecuencia [Hz]')
ax4.set_xlabel('tiempo [s]')
ax4.set_title('Un "Hey!"')

fig.tight_layout()
plt.show()


# %matplotlib inline

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

'''
Los valores obtenidos de amplitud, para realizar mediciones reales, todavía requeren ser convertidos a nivel de presión sonora.
Las gráficas resultantes en esta celda solo tienen el propósito de demostración
'''

x_cum = integrate.cumtrapz( np.power(np.flip(x[:,0]), 2))       #se realiza integral acumulativa del cuadrado de la senal invertida en el tiempo
time = np.linspace(0, len(x_cum) / 44100, len(x_cum))           #array de tiempo

#se vuelve a invertir la senal en el tiempo
ax1.plot(time, np.flip(x_cum), 'r')
ax1.set_ylabel('amplitud')
ax1.set_xlabel('tiempo [s]')
ax1.set_title('Eco fluctuante')

y_cum = integrate.cumtrapz( np.power(np.flip(y[:,0]), 2))
time = np.linspace(0, len(y_cum) / 44100, len(y_cum))

ax2.plot(time, np.flip(y_cum), 'b')
ax2.set_ylabel('amplitud')
ax2.set_xlabel('tiempo [s]')
ax2.set_title('Recinto reverberante')

z_cum = integrate.cumtrapz( np.power(np.flip(z[:,0]), 2))
time = np.linspace(0, len(z_cum) / 44100, len(z_cum))

ax3.plot(time, np.flip(z_cum), 'g')
ax3.set_ylabel('amplitud')
ax3.set_xlabel('tiempo [s]')
ax3.set_title('Multiples impulsos')
#ax3.set_xlim((0,0.55))

a_cum = integrate.cumtrapz( np.power(np.flip(a), 2))
time = np.linspace(0, len(a_cum) / 44100, len(a_cum))

ax4.plot(time, np.flip(a_cum), 'k')
ax4.set_ylabel('amplitud')
ax4.set_xlabel('tiempo [s]')
ax4.set_title('Un "Hey!"')

fig.tight_layout()
plt.show()

# %matplotlib inline

fs = 44100  # frecuencia de muestreo
seconds = 2  # duración de la grabación

'''Este notebook no permite la grabación por medio de sounddevice, 
pero las siguientes tres lineas se pueden descomentar para su uso en un notebook regular '''
#myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
#sd.wait()  # Wait until recording is finished
#write('voice.wav', fs, myrecording)  # Save as WAV file 

data, samplerate = sf.read('/content/voice.wav')               # lectura de archivo de grabación de voz
l = len(data)
t = np.linspace(0, l / samplerate, l)
v = data / np.max(np.abs(data))

plt.plot(t, v, 'k')
plt.title("")
plt.show()

# se saltan los datos no necesarios en la IR (los que son 0 ants de que inicie el impulso)
conv = signal.convolve(y[25000:,0], v, mode="full")
l = len(conv)
t = np.linspace(0, l / fs, l)
#conv = conv / np.max([np.max(conv),-np.min(conv)])

scaled = np.int16(conv/np.max(np.abs(conv)) * 32767)    # se normaliza el array y se convierte a escala de profundidan en bits
plt.plot(t, conv)
plt.show()
write('/content/convolution.wav', fs, scaled)           # escritura de archivo .wav

'''Al igual que la grabación, la reproducción por medio de playsound no es posible en este notebook.
  En este caso se creó el componente html para reproducción.
'''
Audio('/content/voice.wav', autoplay=False)
#playsound('/content/voice.wav')

Audio('/content/convolution.wav', autoplay=False)
#playsound('/content/convolution.wav')

''' para evitar divisiones por 0, se le agrega offset al impulso a la hora de realizar una deconvolución
 esto implica que se estan agregando características nuevas al impulso, y por lo tanto la senal resultante
 deberá ser tratada para regresar a su estado original (filtros)
''' 
# la operación regresa la senal original mas un residuo que le es sumado para completar una convolución nueva
deconv, remain = signal.deconvolve(conv, y[25000:, 0] + 100)
l = len(deconv)
t = np.linspace(0, l / fs, l)
deconv = deconv / np.max(np.abs(deconv))

cutoff = 6000 #frecuencia de corte del filtro
order = 12 #orden del filtro
nyq = 0.5 * fs #máxima frecuencia
normal_cutoff = cutoff / nyq #calculo real de frecuencia de corte
b, a = butter(order, normal_cutoff, btype='low', analog=False) #generacion de numerador y denominador del modelo matematico del filtro
print("Numerador: ", b)
print("Denominador: ", a)

filtered = lfilter(b, a, deconv) #senal con el filtro aplicado

scaled = np.int16(filtered/np.max(np.abs(filtered)) * 32767)
print(np.shape(deconv), np.shape(remain))
plt.plot(deconv)
plt.show()

write('/content/deconvolution.wav', fs, scaled)

Audio('/content/deconvolution.wav', autoplay=False)
#playsound('deconvolution.wav')