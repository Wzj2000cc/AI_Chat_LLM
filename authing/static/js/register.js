//注册
function doRegister(){
    var account=$("#account").val();
    var pwd=$("#pwd").val();
    var pwdAgain=$("#pwdAgain").val();
    var email=$("#email").val();
    if(account==''){
        layer.msg('账号不能为空');
        return;
    }
     if(pwd==''){
        layer.msg('密码不能为空');
        return;
    }
     if(pwd!=pwdAgain){
        layer.msg('两次密码输入不一致');
        return;
    }
     $.ajax({
         headers: {
        'X-CSRFToken': $('input[name="csrfmiddlewaretoken"]').val()
    },
        //请求方式
        type : "POST",
        //请求地址
        url : "./doRegister",
        //数据，json字符串
        data : {account:account,pwd:pwd,email:email},
        //请求成功
        success : function(resstr) {
            res=JSON.parse(resstr)
            if(res.code==200){
                layer.msg('注册成功');
                setTimeout(function(){
                   location.href="./admin/home"
                },1500)
            }else{
                layer.msg('账号已存在，请重试');
            }
        },
        //请求失败，包含具体的错误信息
        error : function(e){
            layer.msg('系统错误，稍后请重试');
        }
    });
}