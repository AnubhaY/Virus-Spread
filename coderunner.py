import virusSpread

if __name__=="__main__":

    algorithm = 'matrix' # algorithms = matrix (faster) or grid (graph)
    x,y = input().split()
    data = []
    
    for i in range(int(x)):
        data.append(input().split())
        
    matrix = virusSpread.MatrixGenerator.generate(data, algo=algorithm)
    print(f'{matrix.timer_till_all_infected()}')
