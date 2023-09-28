#include <iostream>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <random>
#include <tuple>

using namespace std;

default_random_engine generator(time(0));
cauchy_distribution<double> distrib(0.0, 1.0); //柯西分布
//normal_distribution<double> distrib(0.0, 1.0);  //高斯分布


/**
 * @brief fitness function
 * @param x    输入参数x
 * @param y    输入参数y
 *
 * @return z   fitness function在(x,y)处的值
 */
double fitness_function(double x, double y)
{
    //double z = 0.9 * exp((-(x + 5) * (x + 5) - (y + 5) * (y + 5)) / 10) + 0.99996 * exp((-(x - 5) * (x - 5) - (y - 5) * (y - 5)) / 20);
    double z = (6.452 * (x + 0.125 * y) * (cos(x) - cos(2 * y)) * (cos(x) - cos(2 * y))) / sqrt(0.8 + (x - 4.2) * (x - 4.2) + 2 * (y - 7) * (y - 7)) + 3.226 * y;
    return z;
    
}

/**
 * @brief 实现遗传算法求解二元函数最值
 */
class GA
{
public:
    int dna_size = 26; //DNA编码长度
    int population_size = 500; //种群大小
    int generation_num = 100; //世代数目
    double cross_rate = 0.8; //交叉率
    double variate_rate = 0.01;//变异率
    int x_low = -10;//x范围下限
    int x_high = 10;//x范围上限
    int y_low = -10;//y范围下限
    int y_high = 10;//y范围上限
    double (*pf)(double x, double y) = fitness_function; //函数指针，读取fitness function
    int** population_matrix = NULL;//二维数组，用于存储当前种群各个member的DNA
    int** tmp_matrix = NULL;//二维数组，用于复制population_matrix
    double* x_vector = NULL;//一维数组，用于存储种群代表的坐标x向量
    double* y_vector = NULL;//二维数组，用于存储种群代表的坐标x向量
    double* fitness_vector = NULL;//一维数组，用于存储当前种群各个member的适应度
    int* decode_vector = NULL;//一维数组，存储用于解码的一维向量
    int** x_matrix = NULL;//二维数组，用于存储当前种群各个member的X DNA
    int** y_matrix = NULL;//二维数组，用于存储当前种群各个member的Y DNA

    int* bestDNA = NULL;

    char variate_flag = 'f';

    GA(int ds, int ps, int gn, double cr, double vr, int xl, int xh, int yl, int yh, char vf);
    void getFunction(double (*point_of_function)(double, double));
    pair<double*, double*> decodeDNA(int** population_matrix);
    int* crossDNA(int* childDNA, int** population_matrix);
    int* variateDNA(int* childDNA);
    double* getFitnessVector(int** population_matrix);
    int** naturalSelect(int** population_matrix, double* fitness_vector);
    int** updatePopulation(int** population_matrix);
    tuple<double, double, double> getResult(int** population_matrix);
    void addSuperDNA(int** population_matrix);
};

/**
 * @brief 类初始化函数
 * @param    输入初始化参数
 * @param y    输入参数y
 *
 * @return z   fitness function在(x,y)处的值
 */
GA::GA(int ds, int ps, int gn, double cr, double vr, int xl, int xh, int yl, int yh, char vf)
{
    dna_size = ds; //DNA编码长度
    population_size = ps; //种群大小
    generation_num = gn; //世代数目
    cross_rate = cr; //交叉率
    variate_rate = vr;//变异率
    x_low = xl;//x范围下限
    x_high = xh;//x范围上限
    y_low = yl;//y范围下限
    y_high = yh;//y范围上限
    variate_flag = vf;

    //为二维数组population_matrix开辟空间
    population_matrix = new int* [population_size]; //开辟指针数组
    for (int i = 0; i < population_size; i++)
        population_matrix[i] = new int[dna_size * 2];

    //为二维数组tmp_matrix开辟空间
    tmp_matrix = new int* [population_size]; //开辟指针数组
    for (int i = 0; i < population_size; i++)
        tmp_matrix[i] = new int[dna_size * 2];

    //生成随机种群矩阵
    for (int i = 0; i < population_size; ++i)
    {
        for (int j = 0; j < dna_size * 2; ++j)
        {
            population_matrix[i][j] = rand() % 2;
        }
    }

    //解码向量，用于二进制转十进制，其值为[2^25 2^24 ... 2^1 2^0]
    decode_vector = new int[dna_size];
    for (int i = 0; i < dna_size; i++)
        decode_vector[i] = pow(2, dna_size - 1 - i);

    //为二维数组x_matrix开辟空间
    x_matrix = new int* [population_size]; //开辟指针数组
    for (int i = 0; i < population_size; i++)
        x_matrix[i] = new int[dna_size];

    //为二维数组y_matrix开辟空间
    y_matrix = new int* [population_size]; //开辟指针数组
    for (int i = 0; i < population_size; i++)
        y_matrix[i] = new int[dna_size];

    //为一维数组开辟空间
    fitness_vector = new double[population_size];
    x_vector = new double[population_size];
    y_vector = new double[population_size];
    bestDNA = new int[dna_size * 2];

    //bestDNA初始化
    for (int j = 0; j < dna_size * 2; j++)
        bestDNA[j] = 0;
}

/**
 * @brief 获得fitness function的函数指针
 * @param point_of_function 函数指针
 */
void GA::getFunction(double (*point_of_function)(double, double))
{
    pf = point_of_function;
}

/**
 * @brief 对DNA进行解码
 * @param population_matrix 种群矩阵
 *
 * @return x_vector, y_vector   x和y的坐标向量
 */
pair<double*, double*> GA::decodeDNA(int** population_matrix)
{
    //矩阵分割，取前一半列的矩阵
    for (int i = 0; i < population_size; i++)
        for (int j = 0; j < dna_size; j++)
            x_matrix[i][j] = population_matrix[i][j];

    //矩阵分割，取后一半列的矩阵
    for (int i = 0; i < population_size; i++)
        for (int j = 0; j < dna_size; j++)
            y_matrix[i][j] = population_matrix[i][j + dna_size];

    //种群x向量，由二进制转换成十进制并映射到x区间
    for (int i = 0; i < population_size; i++)
    {
        x_vector[i] = 0;
        for (int j = 0; j < dna_size; j++)
            x_vector[i] += x_matrix[i][j] * decode_vector[j];
        x_vector[i] = x_vector[i] / (pow(2, dna_size) - 1) * (x_high - x_low) + x_low;
    }
    //种群y向量，由二进制转换成十进制并映射到x区间
    for (int i = 0; i < population_size; i++)
    {
        y_vector[i] = 0;
        for (int j = 0; j < dna_size; j++)
            y_vector[i] += y_matrix[i][j] * decode_vector[j];
        y_vector[i] = y_vector[i] / (pow(2, dna_size) - 1) * (y_high - y_low) + y_low;
    }

    //通过pair返回两个值
    return make_pair(x_vector, y_vector);
}

/**
 * @brief DNA交叉，父DNA随机与一个母DNA在随机位置发生DNA交叉
 * @param childDNA    继承父DNA的子DNA
 * @param tmp_matrix  原始种群的DNA集
 *
 * @return childDNA   DNA交叉后产生的子DNA
 */
int* GA::crossDNA(int* childDNA, int** tmp_matrix)
{
    if (rand()/RAND_MAX < cross_rate) //以一定的概率发生交叉
    {
        int mother_random = rand() % population_size; //随机选择一个母DNA
        int* motherDNA = tmp_matrix[mother_random];
        int cross_position = rand() % (dna_size * 2); //随机选择一个交叉位置
        for (int i = cross_position; i < dna_size * 2; i++)
            childDNA[i] = motherDNA[i];  //得到交叉以后的子DNA
    }
    return childDNA;
}

/**
 * @brief DNA变异，子DNA在随机位置产生变异（位翻转/柯西变异）
 * @param childDNA    子DNA
 *              variate_flag为f时，位翻转变异
 *              variate_flag为d时，柯西变异
 *
 * @return childDNA   DNA变异后的子DNA
 */
int* GA::variateDNA(int* childDNA)
{
    if (variate_flag == 'f') //启用位翻转变异
    {
        if (rand() / RAND_MAX < variate_rate) //以一定概率发生变异
        {
            int variate_position = rand() % (dna_size * 2); //随机一个位置
            if (childDNA[variate_position] == 0)
                childDNA[variate_position] = 1;
            else
                childDNA[variate_position] = 0;
        } //进行位翻转
    }
    else if (variate_flag == 'd') //启用柯西变异
    {
        if (rand() / RAND_MAX < variate_rate) //以一定概率发生变异
        {
            //解码x,y坐标值
            int x = 0;
            for (int i = 0; i < dna_size; i++)
                x += childDNA[i] * decode_vector[i];
            int y = 0;
            for (int i = 0; i < dna_size; i++)
                y += childDNA[i] * decode_vector[i];

            //x坐标柯西变异
            x = x * (1 + distrib(generator));
            if (x > (pow(2, dna_size) - 1))
                x = (pow(2, dna_size) - 1);
            else if (x < 0)
                x = 0;

            //y坐标柯西变异
            y = y * (1 + distrib(generator));
            if (y > (pow(2, dna_size) - 1))
                y = (pow(2, dna_size) - 1);
            else if (y < 0)
                y = 0;

            //十进制转二进制，转回到childDNA
            for (int i = 1; i <= dna_size; i++)
            {
                if (x > 0)
                {
                    childDNA[dna_size - i] = x % 2;
                    x = (int)(x / 2);
                }
                else
                {
                    childDNA[dna_size - i] = 0;
                }
            }

            for (int i = 1; i <= dna_size; i++)
            {
                if (y > 0)
                {
                    childDNA[dna_size - i + dna_size] = y % 2;
                    y = (int)(y / 2);
                }
                else
                {
                    childDNA[dna_size - i + dna_size] = 0;
                }
            }
        }
    }

    return childDNA;
}

/**
 * @brief 计算适应度向量，即每个个体的适应度
 * @param population_matrix 种群矩阵
 *
 * @return fitness_vector  适应度向量
 */
double* GA::getFitnessVector(int** population_matrix)
{
    pair<double*, double*> population_vector = decodeDNA(population_matrix);
    x_vector = population_vector.first;
    y_vector = population_vector.second;//解码DNA

    for (int i = 0; i < population_size; i++)
        fitness_vector[i] = (*pf)(x_vector[i], y_vector[i]); //计算适应度
    double minValue = *min_element(fitness_vector, fitness_vector + population_size);//获得数组的最小值
    for (int i = 0; i < population_size; i++)
        fitness_vector[i] = fitness_vector[i] - minValue + 0.001;
    return fitness_vector;
}

/**
 * @brief 自然选择，采用轮盘赌方法
 * @param population_matrix 种群矩阵
 * fitness_vector 适应度向量
 *
 * @return fitness_vector  自然选择后的种群矩阵
 */
int** GA::naturalSelect(int** population_matrix, double* fitness_vector)
{
    //复制population_matrix
    for (int i = 0; i < population_size; ++i)
    {
        for (int j = 0; j < dna_size * 2; ++j)
        {
            tmp_matrix[i][j] = population_matrix[i][j];
        }
    }

    //对适应度向量求和
    double sum_fitness_vector = 0.0;
    for (int j = 0; j < population_size; j++)
        sum_fitness_vector += fitness_vector[j];

    //更新population_matrix
    for (int i = 0; i < population_size; i++)
    {
        //按照fitness_vector概率选择
        double randdouble = rand() / double(RAND_MAX) * sum_fitness_vector;
        double curValue = 0.0;
        for (int j = 0; j < population_size; j++)
        {
            curValue += fitness_vector[j];
            if (randdouble < curValue)
            {
                //选中下标j，将tmp_matrix的第j行复制到population_matrix的第i行
                for (int k = 0; k < dna_size * 2; k++)
                    population_matrix[i][k] = tmp_matrix[j][k];
                break;
            }
        }
    }
    return population_matrix;
}

/**
 * @brief 更新种群
 * @param population_matrix 种群矩阵
 *
 * @return population_matrix  更新后的种群矩阵
 */
int** GA::updatePopulation(int** population_matrix)
{
    //复制population_matrix
    for (int i = 0; i < population_size; ++i)
    {
        for (int j = 0; j < dna_size * 2; ++j)
        {
            tmp_matrix[i][j] = population_matrix[i][j];
        }
    }

    for (int i = 0; i < population_size; i++)
    {
        int* childDNA = tmp_matrix[i]; //子DNA继承父DNA
        childDNA = crossDNA(childDNA, tmp_matrix);
        childDNA = variateDNA(childDNA);
        //更新种群
        for (int j = 0; j < dna_size * 2; j++)
            population_matrix[i][j] = childDNA[j];
    }
    return population_matrix;
}

/**
 * @brief 获得当前种群中的最优个体
 * @param population_matrix 种群矩阵
 *
 * @return x,y,maxValue  最优个体代表的x,y坐标和函数最大值
 */
tuple<double, double, double> GA::getResult(int** population_matrix)
{
    fitness_vector = getFitnessVector(population_matrix); //计算适应度向量
    int maxPosition = max_element(fitness_vector, fitness_vector + population_size) - fitness_vector; //最大适应度索引
    pair<double*, double*> population_vector = decodeDNA(population_matrix);
    x_vector = population_vector.first; //获得x值序列
    y_vector = population_vector.second; //获得y值序列
    double maxValue = (*pf)(x_vector[maxPosition], y_vector[maxPosition]); //获得最大值

    for (int j = 0; j < dna_size * 2; j++)
        bestDNA[j] = population_matrix[maxPosition][j]; //记录最好的个体

    //cout << "x:" << x_vector[maxPosition] << endl;
    //cout << "y:" << y_vector[maxPosition] << endl;
    //cout << "maxValue:" << maxValue << endl;
    cout << x_vector[maxPosition] << "," << y_vector[maxPosition] << "," << maxValue << endl;

    return make_tuple(x_vector[maxPosition], y_vector[maxPosition], maxValue);
}

/**
 * @brief 向群体中加入超级个体
 * @param population_matrix 种群矩阵
 *
 */
void GA::addSuperDNA(int** population_matrix)
{
    int index_random = rand() % population_size; //随机选择一个替换DNA
    for (int j = 0; j < dna_size * 2; j++)
        population_matrix[index_random][j] = bestDNA[j]; //将超级个体加入种群
}

//主函数
int main()
{
    cout << "Begin...\n" << endl;
    double (*point_of_function)(double, double) = fitness_function;
    tuple<double, double, double> result;
    double current_x = 0, current_y = 0, current_max = -1000, maxx = 0, maxy = 0, maxValue = -1000;

    // fitness function 1
    //GA ga(30, 100, 200, 0.7, 0.15, -10, 10, -10, 10, 'f'); //位翻转
    //GA ga(30, 200, 400, 0.7, 0.1, -10, 10, -10, 10, 'd'); //柯西分布
    
    //fitness function 2
    //GA ga(30, 100, 200, 0.7, 0.2, 0, 10, 0, 10, 'f'); //位翻转
    GA ga(30, 200, 300, 0.6, 0.09, 0, 10, 0, 10, 'd'); //柯西分布

    ga.getFunction(point_of_function); //获得函数指针

    srand((unsigned)time(NULL));

    //种群演化迭代
    for (int r = 0; r < ga.generation_num; r++)
    {
        //cout << "Round " << (r+1) << endl;
        ga.population_matrix = ga.updatePopulation(ga.population_matrix);
        ga.fitness_vector = ga.getFitnessVector(ga.population_matrix);
        ga.population_matrix = ga.naturalSelect(ga.population_matrix, ga.fitness_vector);
        ga.addSuperDNA(ga.population_matrix);
        result = ga.getResult(ga.population_matrix);
        
        /*tie(current_x, current_y, current_max) = result;
        if (current_max > maxValue)
        {
            maxx = current_x;
            maxy = current_y;
            maxValue = current_max;
        }*/

        //cout << endl;
    }
    cout << "End..." << endl;
    cout << "-----------------------------" << endl;
    cout << "GA result:" << endl;

    //cout << "x:" << maxx << endl;
    //cout << "y:" << maxy << endl;
    //cout << "maxValue:" << maxValue << endl;
    
    tie(current_x, current_y, current_max) = result;
    cout << "x:" << current_x << endl;
    cout << "y:" << current_y << endl;
    cout << "maxValue:" << current_max << endl;

    system("pause");
}
