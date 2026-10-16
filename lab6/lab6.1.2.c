#include <stdio.h>

enum { N = 5, K = 3, DELTA = K / 2 };

void conv2d(const double X[N][N], const double V[K][K],
            double u, double H[N][N]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            H[i][j] = u;

            for (int a = -DELTA; a <= DELTA; ++a) {
                for (int b = -DELTA; b <= DELTA; ++b) {
                    int row = i + a;
                    int col = j + b;

                    if (row >= 0 && row < N && col >= 0 && col < N) {
                        H[i][j] += V[a + DELTA][b + DELTA] * X[row][col];
                    }
                }
            }
        }
    }
}

void print_matrix(const double matrix[N][N]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%5.0f", matrix[i][j]);
        }
        putchar('\n');
    }
}

int main(void) {
    double X[N][N] = {
        { 1,  2,  3,  4,  5},
        { 6,  7,  8,  9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25}
    };
    const double V[K][K] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };
    double u = 1;
    double H[N][N];

    printf("X:\n");
    print_matrix(X);
    printf("\nV = 3x3 ones, u = %.0f\n", u);

    conv2d(X, V, u, H);
    double before = H[2][2];
    printf("H[2][2] = 1 + 7 + 8 + 9 + 12 + 13 + 14 + 17 + 18 + 19 = %.0f\n",
           before);

    X[0][0] = 10000;
    conv2d(X, V, u, H);
    printf("\n把远处的 X[0][0] 改成 10000 后，H[2][2] = %.0f\n", H[2][2]);
    if (before != 118 || H[2][2] != before) {
        return 1;
    }

    double X_left[N][N] = {{0}};
    double X_right[N][N] = {{0}};
    double H_left[N][N];
    double H_right[N][N];
    X_left[2][1] = 1;
    X_right[2][2] = 1;

    conv2d(X_left, V, 0, H_left);
    conv2d(X_right, V, 0, H_right);

    printf("\n下面两个例子共用同一个 V，偏置设为 0。\n");
    printf("只有 X[2][1] = 1 时，输出 H：\n");
    print_matrix(H_left);
    printf("\n把这个输入点右移一格到 X[2][2] 后，输出 H：\n");
    print_matrix(H_right);

    for (int i = 0; i < N; ++i) {
        for (int j = 1; j < N; ++j) {
            if (H_right[i][j] != H_left[i][j - 1]) {
                return 1;
            }
        }
    }
    printf("\n该例中，输出中的响应也向右移动了一格。\n");
    return 0;
}
