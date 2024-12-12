"""
Resolver as equações do fluxo de fluidos usando o método "Stable Fluids" de Jos Stam com a FFT para obter simulações ultra-rápidas. Estende a versão 2D deste código para 3D.

Momento:           ∂u/∂t + (u ⋅ ∇) u = − 1/ρ ∇p + ν ∇²u + f

Incompressibilidade:  ∇ ⋅ u = 0

u:  Velocidade (vetor 3D)  
p:  Pressão  
f:  Força  
ν:  Viscosidade Cinética  
ρ:  Densidade  
t:  Tempo  
∇:  Operador Nabla (definindo convecção não-linear, gradiente e divergência)  
∇²: Operador de Laplace  

----

Um domínio em forma de cubo unitário com Condições de Contorno Periódicas (por exemplo, o que sai pelo topo entra novamente pela base).

           +--------+  
          /        /|  
         /        / |  
        +--------+  |  
        |        |  |  
        |        |  +  
        |        | /  
        |        |/  
        +--------+  

-> Dois patches de força em direções opostas, centralizados verticalmente, mas ligeiramente deslocados.

-----

Estratégia de Solução:

-> Começar com velocidade zero em todos os lugares: u = [0, 0, 0]

1. Adicionar forças

    w₁ = u + Δt f

2. Convectar por auto-advecção (definir o valor na localização atual como sendo o valor na posição rastreada no fluxo de corrente.) -> incondicionalmente estável

    w₂ = w₁(p(x, −Δt))

3. Difundir e Projetar no Domínio de Fourier

    3.1 Transformação para o Domínio de Fourier  

        w₂ → w₃
    
    3.2 Difundir por "filtragem de baixa frequência" (convolução é multiplicação no Domínio de Fourier)  

        w₄ = exp(− k² ν Δt) w₃
    
    3.3 Calcular a (pseudo-)pressão no Domínio de Fourier avaliando a divergência no Domínio de Fourier  

        q = w₄ ⋅ k / ||k||₂
    
    3.4 Corrigir as velocidades para que sejam incompressíveis  

        w₅ = w₄ − q k / ||k||₂
    
    3.5 Transformação Inversa de volta ao domínio espacial  

        w₆ ← w₅

4. Repetir

k = [k_x, k_y, k_z] são as frequências espaciais (= números de onda).

A Transformada de Fourier prescreve implicitamente as Condições de Contorno Periódicas.

-------

Mudanças em relação ao vídeo original (https://youtu.be/bvPi6XwdM0U)

1. Alterar o pré-fator de tempo na força para aplicá-lo por mais tempo (veja o passo (1) no loop temporal).

2. Usar "clamping" periódico e aumentar o número de passos de tempo.

5. Grupo - Mudança e Configuração em relação ao video original - Andre Moura e Victor Coelho.
"""


# Importa pacotes necessários:
# - FFTW: biblioteca para realizar transformadas rápidas de Fourier (Fast Fourier Transform - FFT)
# - WriteVTK: permite salvar dados em arquivos no formato VTK para visualização
# - ProgressMeter: exibe uma barra de progresso para loops
# - Interpolations: fornece métodos para interpolação em várias dimensões
# - LinearAlgebra: inclui operações de álgebra linear como norm e transformações matriciais

using FFTW
using WriteVTK
using ProgressMeter
using Interpolations
using LinearAlgebra

# Define constantes que configuram o domínio e a simulação
N_POINTS = 42               # Número de pontos em cada dimensão da malha cúbica. Onde o autor usou 44 pontos, e nós utilizamos 42 neste exemplo (mesh)
KINEMATIC_VISCOSITY = 0.0001 # Viscosidade cinemática 
TIME_STEP_LENGTH = 0.02      # Comprimento do passo de tempo, aumentado para 0.02 em comparação do autor, que usou 0.01.
N_TIME_STEPS = 150           # 100/ Número total de passos de tempo
# Função para retroceder posições no campo vetorial
function backtrace!(backtraced_positions, original_positions, direction)
    # Move as partículas "para trás no tempo" ao longo de uma direção, considerando o passo de tempo
    # Utiliza mod1 para manter as posições dentro do intervalo [0, 1] (condições periódicas)
    backtraced_positions[:] = mod1.(
        original_positions - TIME_STEP_LENGTH * direction,
        1.0
    )
end

# Função para interpolar valores em posições específicas dentro de um campo
function interpolate_positions!(
    field_interpolated,
    field,
    interval_x,
    interval_y,
    interval_z,
    query_points_x,
    query_points_y,
    query_points_z,
)
    interpolator = LinearInterpolation(
        (interval_x, interval_y, interval_z),
        field,
    )
    field_interpolated[:] = interpolator.(query_points_x, query_points_y, query_points_z)
end

# Função principal que implementa a simulação
function main()
    element_length = 1.0 / (N_POINTS - 1) # Quanto maior o N_POINTS, menor o elemento MESH.
    x_interval = 0.0:element_length:1.0
    y_interval = 0.0:element_length:1.0
    z_interval = 0.0:element_length:1.0
# Semelhante ao meshgrid no NumPy  /
    #Essas variáveis representam as coordenadas x, 𝑦 e z de todos os pontos na malha 3D.
    coordinates_x = [x for x in x_interval, y in y_interval, z in z_interval]
    coordinates_y = [y for x in x_interval, y in y_interval, z in z_interval]
    coordinates_z = [z for x in x_interval, y in y_interval, z in z_interval]

    wavenumbers_1d = fftfreq(N_POINTS) .* N_POINTS

    wavenumbers_x = [k_x for k_x in wavenumbers_1d, k_y in wavenumbers_1d, k_z in wavenumbers_1d]
    wavenumbers_y = [k_y for k_x in wavenumbers_1d, k_y in wavenumbers_1d, k_z in wavenumbers_1d]
    wavenumbers_z = [k_z for k_x in wavenumbers_1d, k_y in wavenumbers_1d, k_z in wavenumbers_1d]
    wavenumbers_norm = [norm([k_x, k_y, k_z]) for k_x in wavenumbers_1d, k_y in wavenumbers_1d, k_z in wavenumbers_1d]

    decay = exp.(- TIME_STEP_LENGTH .* KINEMATIC_VISCOSITY .* wavenumbers_norm.^2)

    wavenumbers_norm[iszero.(wavenumbers_norm)] .= 1.0
    normalized_wavenumbers_x = wavenumbers_x ./ wavenumbers_norm
    normalized_wavenumbers_y = wavenumbers_y ./ wavenumbers_norm
    normalized_wavenumbers_z = wavenumbers_z ./ wavenumbers_norm

    # Defina as forças

    force_x = 100.0 .* (
        ifelse.(
            (coordinates_x .> 0.2)
            .&
            (coordinates_x .< 0.3)
            .&
            (coordinates_y .> 0.45)
            .&
            (coordinates_y .< 0.52)
            .&
            (coordinates_z .> 0.45)
            .&
            (coordinates_z .< 0.52),
            1.0,
            0.0,
        )
        -
        ifelse.(
            (coordinates_x .> 0.7)
            .&
            (coordinates_x .< 0.8)
            .&
            (coordinates_y .> 0.48)
            .&
            (coordinates_y .< 0.55)
            .&
            (coordinates_z .> 0.48)
            .&
            (coordinates_z .< 0.55),
            1.0,
            0.0,

        )
    )

    # Pré-alocar todos os arrays

    backtraced_coordinates_x = zero(coordinates_x)
    backtraced_coordinates_y = zero(coordinates_y)
    backtraced_coordinates_z = zero(coordinates_z)

    velocity_x = zero(coordinates_x)
    velocity_y = zero(coordinates_y)
    velocity_z = zero(coordinates_z)

    velocity_x_prev = zero(velocity_x)
    velocity_y_prev = zero(velocity_y)
    velocity_z_prev = zero(velocity_z)

    velocity_x_fft = zero(velocity_x)
    velocity_y_fft = zero(velocity_y)
    velocity_z_fft = zero(velocity_z)
    pressure_fft = zero(coordinates_x)

    velocity_x_trajectory = []
    velocity_y_trajectory = []
    velocity_z_trajectory = []

    @showprogress "Timestepping ..." for iter in 1:N_TIME_STEPS
        # (1) Aplique as forças
        time_current = (iter - 1) * TIME_STEP_LENGTH
        pre_factor = max(1 - time_current, 0.0)
        velocity_x_prev += TIME_STEP_LENGTH * pre_factor * force_x

        # (2) Autoadvecção por retrocesso e interpolação


        backtrace!(backtraced_coordinates_x, coordinates_x, velocity_x_prev)
        backtrace!(backtraced_coordinates_y, coordinates_y, velocity_y_prev)
        backtrace!(backtraced_coordinates_z, coordinates_z, velocity_z_prev)
        interpolate_positions!(
            velocity_x,
            velocity_x_prev,
            x_interval,
            y_interval,
            z_interval,
            backtraced_coordinates_x,
            backtraced_coordinates_y,
            backtraced_coordinates_z,
        )
        interpolate_positions!(
            velocity_y,
            velocity_y_prev,
            x_interval,
            y_interval,
            z_interval,
            backtraced_coordinates_x,
            backtraced_coordinates_y,
            backtraced_coordinates_z,
        )
        interpolate_positions!(
            velocity_z,
            velocity_z_prev,
            x_interval,
            y_interval,
            z_interval,
            backtraced_coordinates_x,
            backtraced_coordinates_y,
            backtraced_coordinates_z,
        )

        # (3.1) Transformada de Fourier no dominio
        velocity_x_fft = fft(velocity_x)
        velocity_y_fft = fft(velocity_y)
        velocity_z_fft = fft(velocity_z)
        # (3.2) Autoadvecção por retrocesso e interpolação

        velocity_x_fft .*= decay
        velocity_y_fft .*= decay
        velocity_z_fft .*= decay



# (3.3) Calcular Pseudo-Pressão por Divergência no Domínio de Fourier
        pressure_fft = (
            velocity_x_fft .* normalized_wavenumbers_x
            +
            velocity_y_fft .* normalized_wavenumbers_y
            +
            velocity_z_fft .* normalized_wavenumbers_z
        )


        # (3.4) Projete as velocidades como incompressíveis

        velocity_x_fft -= pressure_fft .* normalized_wavenumbers_x
        velocity_y_fft -= pressure_fft .* normalized_wavenumbers_y
        velocity_z_fft -= pressure_fft .* normalized_wavenumbers_z


        # (3.5) Transformar novamente em domínio espacial
        velocity_x = real(ifft(velocity_x_fft))
        velocity_y = real(ifft(velocity_y_fft))
        velocity_z = real(ifft(velocity_z_fft))


        # Avançar no tempo

        velocity_x_prev = velocity_x
        velocity_y_prev = velocity_y
        velocity_z_prev = velocity_z


        # Salvar para visualização

        push!(velocity_x_trajectory, velocity_x)
        push!(velocity_y_trajectory, velocity_y)
        push!(velocity_z_trajectory, velocity_z)
    end

    paraview_collection("transient_vector") do pvd
        @showprogress "Writing out to vtk ..." for iter in 1:N_TIME_STEPS
            vtk_grid("timestep_$iter", x_interval, y_interval, z_interval) do vtk
                vtk["velocity"] = (
                    velocity_x_trajectory[iter],
                    velocity_y_trajectory[iter],
                    velocity_z_trajectory[iter],
                )
                time = (iter - 1) * TIME_STEP_LENGTH
                pvd[time] = vtk
            end
        end
    end

end

main()