#define NFLOATS 5
#define NINTS 5

#include "octmps.h"

void Crossproduct(long double x1, long double y1, long double z1,
                  long double x2, long double y2, long double z2,
                  long double *result_x, long double *result_y, long double *result_z) {
    *result_x = y1 * z2 - z1 * y2;
    *result_y = z1 * x2 - x1 * z2;
    *result_z = x1 * y2 - y1 * x2;
}

FLOAT Dotproduct(FLOAT x1, FLOAT y1, FLOAT z1, FLOAT x2, FLOAT y2, FLOAT z2) {
    return x1 * x2 + y1 * y2 + z1 * z2;
}

bool isVerticesSame(Vertex *v1, Vertex *v2) {
    if (v1->x == v2->x && v1->y == v2->y && v1->z == v2->z)
        return true;
    else
        return false;
}

/*********************************************************************
 * Check if two faces are the same but with different vertices orders
 ********************************************************************/
bool IsFacesSame(TriangleFaces *face1, TriangleFaces *face2) {
    int combinations[6][3][2] = {{{0, 0}, {1, 1}, {2, 2}},
                                 {{0, 0}, {1, 2}, {2, 1}},
                                 {{0, 1}, {1, 0}, {2, 2}},
                                 {{0, 1}, {1, 2}, {2, 0}},
                                 {{0, 2}, {1, 1}, {2, 0}},
                                 {{0, 2}, {1, 0}, {2, 1}}};

    int flag;
    for (int j = 0; j < 6; j++) {
        flag = true;
        for (int k = 0; k < 3; k++)
            if (!isVerticesSame(&face1->V[combinations[j][k][0]], &face2->V[combinations[j][k][1]])) {
                flag = false;
                break;
            }
        if (flag)
            return true;
    }

    return flag;
}

int CheckFaceExists(TriangleFaces *faces, int cntr, Vertex *v0, Vertex *v1, Vertex *v2) {
    // This way is much faster than using IsFaceSame function!
    for (int i = 0; i < cntr; i++) {
        if (faces[i].V[0].x == v0->x && faces[i].V[0].y == v0->y && faces[i].V[0].z == v0->z &&
            faces[i].V[1].x == v1->x && faces[i].V[1].y == v1->y && faces[i].V[1].z == v1->z &&
            faces[i].V[2].x == v2->x && faces[i].V[2].y == v2->y && faces[i].V[2].z == v2->z)
            return i;
        if (faces[i].V[0].x == v0->x && faces[i].V[0].y == v0->y && faces[i].V[0].z == v0->z &&
            faces[i].V[2].x == v1->x && faces[i].V[2].y == v1->y && faces[i].V[2].z == v1->z &&
            faces[i].V[1].x == v2->x && faces[i].V[1].y == v2->y && faces[i].V[1].z == v2->z)
            return i;
        if (faces[i].V[1].x == v0->x && faces[i].V[1].y == v0->y && faces[i].V[1].z == v0->z &&
            faces[i].V[0].x == v1->x && faces[i].V[0].y == v1->y && faces[i].V[0].z == v1->z &&
            faces[i].V[2].x == v2->x && faces[i].V[2].y == v2->y && faces[i].V[2].z == v2->z)
            return i;
        if (faces[i].V[1].x == v0->x && faces[i].V[1].y == v0->y && faces[i].V[1].z == v0->z &&
            faces[i].V[2].x == v1->x && faces[i].V[2].y == v1->y && faces[i].V[2].z == v1->z &&
            faces[i].V[0].x == v2->x && faces[i].V[0].y == v2->y && faces[i].V[0].z == v2->z)
            return i;
        if (faces[i].V[2].x == v0->x && faces[i].V[2].y == v0->y && faces[i].V[2].z == v0->z &&
            faces[i].V[0].x == v1->x && faces[i].V[0].y == v1->y && faces[i].V[0].z == v1->z &&
            faces[i].V[1].x == v2->x && faces[i].V[1].y == v2->y && faces[i].V[1].z == v2->z)
            return i;
        if (faces[i].V[2].x == v0->x && faces[i].V[2].y == v0->y && faces[i].V[2].z == v0->z &&
            faces[i].V[1].x == v1->x && faces[i].V[1].y == v1->y && faces[i].V[1].z == v1->z &&
            faces[i].V[0].x == v2->x && faces[i].V[0].y == v2->y && faces[i].V[0].z == v2->z)
            return i;
    }
    return -1;
}


/************************
**  Creating mesh graph
*************************/
TetrahedronStruct *MeshGraph(char *filename, SimulationStruct **simulations) {
    FILE *pFile;
    float ftemp[NFLOATS];
    int itemp[NINTS];

    int number_of_tetrahedrons = 1;
    int number_of_faces = 1;
    int number_of_vertices = 1;
    int i;

    pFile = fopen(filename, "r");
    if (pFile == NULL) {
        perror("Error opening mesh file");
        return 0;
    }

    // First, read the number of vertices
    if (!readints(1, itemp, pFile)) {
        perror("Error reading number of vertices");
        return 0;
    }
    number_of_vertices = itemp[0];

    // Second, read the number of tetrahedrons
    if (!readints(1, itemp, pFile)) {
        perror("Error reading number of tetrahedrons");
        return 0;
    }
    number_of_tetrahedrons = itemp[0];

    FLOAT x, y, z;
    Vertex *vertices = new Vertex[number_of_vertices];
    TetrahedronStruct *tetrahedrons =
            new TetrahedronStruct[number_of_tetrahedrons];

    number_of_faces = 4 * number_of_tetrahedrons;
    TriangleFaces *faces = new TriangleFaces[number_of_faces];

    TetrahedronStruct ***tetrahedrons_adjacency_matrix = new TetrahedronStruct **[number_of_faces];
    for (i = 0; i < number_of_faces; i++)
        tetrahedrons_adjacency_matrix[i] = new TetrahedronStruct *[2];

    for (i = 0; i < number_of_faces; i++) {
        faces[i].idx = i;
        tetrahedrons_adjacency_matrix[i][0] = NULL;
        tetrahedrons_adjacency_matrix[i][1] = NULL;
    }

    for (i = 0; i < number_of_vertices; i++) {
        // Read vertices coordinates (3 x float)
        if (!readfloats(3, ftemp, pFile)) {
            perror("Error reading vertices coordinates, check mesh file");
            return 0;
        }
        vertices[i].idx = i;
        vertices[i].x = ftemp[0];
        vertices[i].y = ftemp[1];
        vertices[i].z = ftemp[2];
    }

    int n_distinct_faces = 0;
    for (i = 0; i < number_of_tetrahedrons; i++) {
        for (int j = 0; j < 4; j++)
            tetrahedrons[i].adjTetrahedrons[j] = NULL;

        if (!readints(5, itemp, pFile)) {
            perror("Error reading tetrahedron points and regions, check the mesh file please!");
            return 0;
        }

        int region = itemp[4];
        for (int j = 0; j < 4; j++)
            tetrahedrons[i].Vertices[j] = vertices[itemp[j] - 1];

        int index_face[4] = {-1, -1, -1, -1};

        int three_combinations[4][3] = {{0, 1, 2},
                                        {0, 1, 3},
                                        {0, 2, 3},
                                        {1, 2, 3}};

        if (i > 0)
            for (int j = 0; j < 4; j++)
                index_face[j] = CheckFaceExists(faces, n_distinct_faces,
                                                &vertices[itemp[three_combinations[j][0]] - 1],
                                                &vertices[itemp[three_combinations[j][1]] - 1],
                                                &vertices[itemp[three_combinations[j][2]] - 1]);

        for (int j = 0; j < 4; j++)
            if (index_face[j] == -1) {
                // New face
                for (int k = 0; k < 3; k++)
                    faces[n_distinct_faces].V[k] = vertices[itemp[three_combinations[j][k]] - 1];

                long double Nx, Ny, Nz;
                Crossproduct(faces[n_distinct_faces].V[0].x - faces[n_distinct_faces].V[1].x,
                             faces[n_distinct_faces].V[0].y - faces[n_distinct_faces].V[1].y,
                             faces[n_distinct_faces].V[0].z - faces[n_distinct_faces].V[1].z,
                             faces[n_distinct_faces].V[0].x - faces[n_distinct_faces].V[2].x,
                             faces[n_distinct_faces].V[0].y - faces[n_distinct_faces].V[2].y,
                             faces[n_distinct_faces].V[0].z - faces[n_distinct_faces].V[2].z, &Nx, &Ny, &Nz);

                FLOAT norm = sqrt(Nx * Nx + Ny * Ny + Nz * Nz);
                Nx /= norm;
                Ny /= norm;
                Nz /= norm;

                faces[n_distinct_faces].Nx = Nx;
                faces[n_distinct_faces].Ny = Ny;
                faces[n_distinct_faces].Nz = Nz;
                faces[n_distinct_faces].d = Dotproduct(-Nx, -Ny, -Nz, faces[n_distinct_faces].V[0].x,
                                                       faces[n_distinct_faces].V[0].y, faces[n_distinct_faces].V[0].z);

                tetrahedrons[i].Faces[j] = faces[n_distinct_faces];
                tetrahedrons_adjacency_matrix[n_distinct_faces][0] = &tetrahedrons[i];
                n_distinct_faces++;
            } else {
                // Face already added to the faces array with index index_face[j]
                tetrahedrons[i].Faces[j] = faces[index_face[j]];
                tetrahedrons[i].adjTetrahedrons[j] = tetrahedrons_adjacency_matrix[index_face[j]][0];
                tetrahedrons_adjacency_matrix[index_face[j]][1] = &tetrahedrons[i];
                for (int k = 0; k < 4; k++)
                    if (IsFacesSame(&tetrahedrons_adjacency_matrix[index_face[j]][0]->Faces[k], &faces[index_face[j]]))
                        tetrahedrons_adjacency_matrix[index_face[j]][0]->adjTetrahedrons[k] = &tetrahedrons[i];
            }

        tetrahedrons[i].index = i;
        tetrahedrons[i].region = region;
    }

    for (i = 0; i < number_of_tetrahedrons; i++) {
        FLOAT midPointx = 0, midPointy = 0, midPointz = 0;
        midPointx += tetrahedrons[i].Vertices[0].x;
        midPointy += tetrahedrons[i].Vertices[0].y;
        midPointz += tetrahedrons[i].Vertices[0].z;

        for (int k = 1; k < 4; k++) {
            midPointx += tetrahedrons[i].Vertices[k].x;
            midPointy += tetrahedrons[i].Vertices[k].y;
            midPointz += tetrahedrons[i].Vertices[k].z;

            midPointx /= 2;
            midPointy /= 2;
            midPointz /= 2;
        }

        long double d;
        for (int j = 0; j < 4; j++) {
            d = (tetrahedrons[i].Faces[j].Nx) * midPointx +
                (tetrahedrons[i].Faces[j].Ny) * midPointy +
                (tetrahedrons[i].Faces[j].Nz) * midPointz +
                tetrahedrons[i].Faces[j].d;
            if (d < 0) {
                tetrahedrons[i].Faces[j].Nx *= -1;
                tetrahedrons[i].Faces[j].Ny *= -1;
                tetrahedrons[i].Faces[j].Nz *= -1;
                tetrahedrons[i].Faces[j].d *= -1;
            }
        }
    }

    (*simulations)->num_tetrahedrons = number_of_tetrahedrons;

    return tetrahedrons;
}

/****************************************
**  Finding the root of the mesh graph.
**  The root is the tetrahedron that the
**  probe resides on one of its faces.
*****************************************/
TetrahedronStruct *FindRootofGraph(TetrahedronStruct *tetrahedrons, SimulationStruct **simulations) {

    int i;
    bool probe_flag = false;
    FLOAT distancesfromplanes[4];

    for (i = 0; i < (*simulations)->num_tetrahedrons; i++) {
        probe_flag = true;
        for (int j = 0; j < 4; j++) {
            distancesfromplanes[j] =
                    Dotproduct((*simulations)->probe_x, (*simulations)->probe_y, (*simulations)->probe_z,
                               tetrahedrons[i].Faces[j].Nx,
                               tetrahedrons[i].Faces[j].Ny, tetrahedrons[i].Faces[j].Nz) + tetrahedrons[i].Faces[j].d;
            if (distancesfromplanes[j] < 0) {
                probe_flag = false;
                break;
            }
        }
        if (probe_flag) {
            for (int j = 0; j < 4; j++)
                if (distancesfromplanes[j] == 0)
                    if (tetrahedrons[i].adjTetrahedrons[j] == NULL)
                        return &tetrahedrons[i];
                    else {
                        perror("The probe is not placed on the surface!!!\n");
                        return NULL;
                    }
        }
    }

    perror("The probe is outside of the medium!!!\n");
    return NULL;
}

/*************************************************
**  Serializing the mesh graph for using in GPU
*************************************************/
void SerializeGraph(TetrahedronStruct *RootTetrahedron, UINT32 number_of_vertices, UINT32 number_of_faces,
                    UINT32 number_of_tetrahedrons,
                    Vertex *d_vertices, TriangleFaces *d_faces, TetrahedronStructGPU *d_root) {

    // n_tetras is the number of tetrahedrons actually used in RootTetrahedron
    int rootIdx = RootTetrahedron[0].index;
    bool seenfaces[number_of_faces];
    for (int ttt = 0; ttt < number_of_faces; ttt++) seenfaces[ttt] = false;
    for (int i = 0; i < number_of_tetrahedrons; i++) {
        d_root[i].idx = RootTetrahedron[i - rootIdx].index;
        d_root[i].region = RootTetrahedron[i - rootIdx].region;
        for (int v = 0; v < 4; v++) {
            d_vertices[RootTetrahedron[i - rootIdx].Vertices[v].idx].idx =
                    RootTetrahedron[i - rootIdx].Vertices[v].idx;
            d_vertices[RootTetrahedron[i - rootIdx].Vertices[v].idx].x =
                    RootTetrahedron[i - rootIdx].Vertices[v].x;
            d_vertices[RootTetrahedron[i - rootIdx].Vertices[v].idx].y =
                    RootTetrahedron[i - rootIdx].Vertices[v].y;
            d_vertices[RootTetrahedron[i - rootIdx].Vertices[v].idx].z =
                    RootTetrahedron[i - rootIdx].Vertices[v].z;
        }
        for (int f = 0; f < 4; f++) {
            d_root[i].faces[f] =
            d_faces[RootTetrahedron[i - rootIdx].Faces[f].idx].idx =
                    RootTetrahedron[i - rootIdx].Faces[f].idx;
            if (!seenfaces[d_root[i].faces[f]]) {
                d_faces[RootTetrahedron[i - rootIdx].Faces[f].idx].d =
                        RootTetrahedron[i - rootIdx].Faces[f].d;
                d_faces[RootTetrahedron[i - rootIdx].Faces[f].idx].Nx =
                        RootTetrahedron[i - rootIdx].Faces[f].Nx;
                d_faces[RootTetrahedron[i - rootIdx].Faces[f].idx].Ny =
                        RootTetrahedron[i - rootIdx].Faces[f].Ny;
                d_faces[RootTetrahedron[i - rootIdx].Faces[f].idx].Nz =
                        RootTetrahedron[i - rootIdx].Faces[f].Nz;
                seenfaces[d_root[i].faces[f]] = true;
            }
            FLOAT midPointx = 0, midPointy = 0, midPointz = 0;
            midPointx += d_vertices[RootTetrahedron[i - rootIdx].Vertices[0].idx].x;
            midPointy += d_vertices[RootTetrahedron[i - rootIdx].Vertices[0].idx].y;
            midPointz += d_vertices[RootTetrahedron[i - rootIdx].Vertices[0].idx].z;
            for (int k = 1; k < 4; k++) {
                midPointx += d_vertices[RootTetrahedron[i - rootIdx].Vertices[k].idx].x;
                midPointy += d_vertices[RootTetrahedron[i - rootIdx].Vertices[k].idx].y;
                midPointz += d_vertices[RootTetrahedron[i - rootIdx].Vertices[k].idx].z;
                midPointx /= 2;
                midPointy /= 2;
                midPointz /= 2;
            }
            long double height;
            height =
                    (d_faces[RootTetrahedron[i - rootIdx].Faces[f].idx].Nx) * midPointx +
                    (d_faces[RootTetrahedron[i - rootIdx].Faces[f].idx].Ny) * midPointy +
                    (d_faces[RootTetrahedron[i - rootIdx].Faces[f].idx].Nz) * midPointz +
                    d_faces[RootTetrahedron[i - rootIdx].Faces[f].idx].d;
            if (height < 0) d_root[i].signs[f] = -1;
            else d_root[i].signs[f] = 1;
        }
        for (int a = 0; a < 4; a++) {
            if (RootTetrahedron[i - rootIdx].adjTetrahedrons[a] == NULL)
                d_root[i].adjTetras[a] = -1; // If not adjacent found
            else
                d_root[i].adjTetras[a] =
                        RootTetrahedron[i - rootIdx].adjTetrahedrons[a]->index;
        }
    }
}
