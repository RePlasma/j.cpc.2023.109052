! roseNNa MLP with ReLU activations, L=30 d=30
! onnx model produced by python file
! inference time with varying batch_size
!
! Óscar Amaro (Mar 2024)
!
! timings @ d=30
! L2  batch_size =         5000 , t=   8.8833333333333352E-006  s, shape:        5000           1
! L10  batch_size =         5000 , t=   3.3880000000000021E-005  s, shape:        5000           1
! L30 batch_size =         5000 , t=   9.6996666666666618E-005  s, shape:        5000           1
! L100  batch_size =         5000 , t=   3.2632666666666604E-004  s, shape:        5000           1
! L200  batch_size =         5000 , t=   6.5334333333333090E-004  s, shape:        5000           1

program L30d30

    USE rosenna
    implicit none
    double precision :: start, finish, elapsed
    integer :: jj, j, batch_size, input_dim = 2, L_layers = 30, d_neurons = 30, Nrepeats=300
    integer, dimension(28) :: bslst = (/1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 75, &
    & 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000 /)
    REAL (c_double), allocatable :: inputs(:,:), outputs(:,:)
    double precision, allocatable :: array_results(:,:)
    integer :: iounit

    ! array to store the results of timing
    allocate( array_results(size(bslst),2) )

    CALL initialize()

    do j = 1, size(bslst)
      batch_size = bslst(j)
      print *, "batch_size=", batch_size
      allocate( inputs(batch_size,2), outputs(batch_size,1) )
      call random_number(inputs)

      ! batch inference
      call cpu_time(start)
      do jj = 1, Nrepeats
        CALL use_model(inputs, outputs)
      end do
      call cpu_time(finish)
      array_results(j,2) = (finish - start)/float(Nrepeats)
      print *, "batch_size = ", batch_size, ", t=", array_results(j,2), " s, shape:", shape(outputs)

      deallocate(inputs, outputs)
    end do


    ! Write results to 2D array
    array_results(:,1) = bslst
    ! Save results to file
    open(newunit=iounit, file='roseNNa_mlp_inferenceTime_batchsize.csv', status='replace', action='write')
    do i = 1, size(bslst)
        do j = 1, 2
            if (j == 2) then
                write(iounit, '(F10.5)', advance='no') array_results(i, j)  ! No comma for last element in a row
            else
                write(iounit, '(F10.5, A)', advance='no') array_results(i, j), ','
            end if
        end do
        write(iounit, *)  ! Newline after each row
    end do
    ! Close file
    close(iounit)


end program L30d30
