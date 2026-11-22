#include <algorithm>
#include <cmath>

// Same alternating log normalizations as confirm.sinkhorn, without fast-math.
extern "C" int sinkhorn_log(double* a, int n, int maximum, double tolerance) {
    auto normalize_row = [&](int i) {
        double m=a[i*n];
        for(int j=1;j<n;j++)m=std::max(m,a[i*n+j]);
        double total=0.;for(int j=0;j<n;j++)total+=std::exp(a[i*n+j]-m);
        double z=m+std::log(total);for(int j=0;j<n;j++)a[i*n+j]-=z;
    };
    int used=0;
    for(int k=0;k<maximum;k++) {
        for(int i=0;i<n;i++)normalize_row(i);
        for(int j=0;j<n;j++) {
            double m=a[j];for(int i=1;i<n;i++)m=std::max(m,a[i*n+j]);
            double total=0.;for(int i=0;i<n;i++)total+=std::exp(a[i*n+j]-m);
            double z=m+std::log(total);for(int i=0;i<n;i++)a[i*n+j]-=z;
        }
        double residual=0.;
        for(int i=0;i<n;i++) {
            double m=a[i*n];for(int j=1;j<n;j++)m=std::max(m,a[i*n+j]);
            double total=0.;for(int j=0;j<n;j++)total+=std::exp(a[i*n+j]-m);
            residual=std::max(residual,std::abs(std::exp(m+std::log(total))-1.));
        }
        used=k+1;if(residual<tolerance)break;
    }
    for(int i=0;i<n;i++)normalize_row(i);
    return used;
}
